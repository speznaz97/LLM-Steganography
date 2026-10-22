import numpy as np
import hashlib
import secrets
from typing import Optional

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from arithmetic import ArithmeticCoder
from utils import np_softmax, stream_cipher, pack_bits, unpack_bits

from syncpool import SyncPool
from discop import Discop
from integrity import crc_bits, check_crc, CRC_BITS

def _make_tail_bit_generator(key, seed_context: list[int]):
    """Generates cryptographically secure, high-entropy pseudo-random bits
    seeded deterministically by the key and the context at the start of the tail."""
    # Ensure key is in bytes format
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key

    step = 0
    byte_buf = b""
    bit_idx = 8
    ctx_bytes = np.asarray(seed_context, np.int64).tobytes()

    def read_bit():
        nonlocal byte_buf, bit_idx, step
        if bit_idx >= 8:
            # Generate deterministic pseudo-random bytes using key_bytes
            h = hashlib.sha256(key_bytes + b"|tail|" + ctx_bytes + step.to_bytes(8, "big")).digest()
            byte_buf = h
            bit_idx = 0
            step += 1
        b = (byte_buf[bit_idx // 8] >> (bit_idx % 8)) & 1
        bit_idx += 1
        return b

    return read_bit

def _make_syncpool(model: LlamaCppModel, cfg: StegoConfig) -> SyncPool:
    return SyncPool(model, key=cfg.crypto_key,
                    temperature=getattr(cfg, "stego_temp", 1.0),
                    top_k=getattr(cfg, "syncpool_topk", 256),
                    prob_floor=getattr(cfg, "syncpool_floor", 1e-5),
                    banned_bytes=tuple(cfg.banned_chars))

def _cover_problem(cover: str, emitted: list[int], sp: SyncPool,
                   cfg: StegoConfig) -> Optional[str]:
    """Why the receiver could NOT re-derive this cover from its text, or None.

    The receiver only ever sees `cover.encode("utf-8")` (after whatever the
    transport does to it) and must match it byte-for-byte against SyncPool
    representatives, so the sender checks that before sending anything."""
    tok_bytes = [sp._token_bytes(t) for t in emitted]
    if any(not b for b in tok_bytes):
        return "emitted a token with no surface bytes (SyncPool fallback / special)"
    if cover.encode("utf-8") != b"".join(tok_bytes):
        return "text does not round-trip to the emitted bytes (split UTF-8 character?)"
    if cover != cover.strip():
        return "leading/trailing whitespace would be trimmed by the transport"
    bad = next((b for b in cfg.banned_chars if b in cover), None)
    if bad is not None:
        return f"banned sequence {bad!r} formed across a token boundary"
    return None

def _embed(messages: list[dict], full_bits: list[int], model: LlamaCppModel,
           cfg: StegoConfig):
    """One generation pass. Returns (cover, problem-or-None, payload_tok, tail_tok)."""
    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    plen   = len(p_ids)
    cur    = p_ids.copy()

    sp = _make_syncpool(model, cfg)
    dc = Discop(key=cfg.crypto_key, precision=32)

    model.reset()
    model.eval(p_ids)
    lg = model.get_logits()

    pos = 0
    def read_bit():
        nonlocal pos
        b = full_bits[pos] if pos < len(full_bits) else 0
        pos += 1
        return b

    step = 0
    while pos < len(full_bits):
        if step > getattr(cfg, "max_gen_tokens", 500):
            raise RuntimeError(
                f"exceeded max_tokens; embedded only "
                f"{pos}/{len(full_bits)} bits — distribution too peaked, "
                f"raise stego_temp / max_gen_tokens")
        if step % 10 == 0:
            print(f"\r  [Gen] {min(pos, len(full_bits))}/{len(full_bits)} bits  {step} tok",
                  end="", flush=True)

        pstep = sp.step(lg, cur)
        cdf   = dc.build_cdf(pstep.masses)
        sym_idx, nb = dc.embed(cdf, cur, read_bit)
        chosen = pstep.reps[sym_idx]
        cur.append(chosen)
        step += 1
        model.eval([chosen])
        lg = model.get_logits()

    tail_bit_reader = _make_tail_bit_generator(cfg.crypto_key, cur)

    tail_step = 0
    tail_min = getattr(cfg, "tail_min", 0)
    tail_max = getattr(cfg, "tail_max", 30)
    sentence_enders = getattr(cfg, "sentence_enders", ['.', '!', '?', '\n'])

    while tail_step < tail_max:
        # Stop as soon as the text ends a sentence -- including right after the
        # payload (tail_min=0), so a cleanly-ended payload gets no tail at all
        if tail_step >= tail_min:
            recent = model.detokenize(cur[-6:]).rstrip()
            last = recent[-1] if recent else ""
            if last in sentence_enders:
                break
            if recent and not any(c.isalnum() for c in recent):
                break

        # Step SyncPool and Discop using the exact same parameters as the payload
        pstep = sp.step(lg, cur)
        cdf   = dc.build_cdf(pstep.masses)

        # Embed secure pseudo-random bits to preserve distribution
        sym_idx, nb = dc.embed(cdf, cur, tail_bit_reader)
        chosen = pstep.reps[sym_idx]

        cur.append(chosen)
        tail_step += 1
        model.eval([chosen])
        lg = model.get_logits()

    cover = model.detokenize(cur[plen:], skip_special=True)
    return cover, _cover_problem(cover, cur[plen:], sp, cfg), step, tail_step

def generate_stego(messages: list[dict], secret: str, model: LlamaCppModel,
                   codec: LLMTextCodec, cfg: StegoConfig) -> str:
    print(f"  Secret: '{secret}'")

    # Zero-header payload with CRC prefix
    raw_bits, _ = codec.encode_bits(secret)
    payload   = crc_bits(secret.encode("utf-8")) + raw_bits
    nonce_bits = getattr(cfg, "nonce_bits", 32)
    print(f"  payload: {len(raw_bits)} body + {CRC_BITS} CRC + {nonce_bits} nonce bits, "
          f"0 length-header bits")

    attempts = max(1, getattr(cfg, "gen_attempts", 3))
    for attempt in range(1, attempts + 1):
        # Fresh random nonce, sent in the clear ahead of the ciphertext: keys the
        # keystream (no two-time pad) and makes every attempt a different cover
        nonce = unpack_bits(secrets.token_bytes((nonce_bits + 7) // 8), nonce_bits)
        full_bits = nonce + stream_cipher(payload, cfg.crypto_key, pack_bits(nonce))

        cover, problem, step, tail_step = _embed(messages, full_bits, model, cfg)
        print(f"\n  Done: {step} tok ({(len(raw_bits) / max(step, 1)):.2f} b/t) + {tail_step} tail tokens")
        if problem is None:
            return cover
        print(f"  [!] attempt {attempt}/{attempts} rejected: {problem}")

    raise RuntimeError(f"no transmissible cover after {attempts} attempts: {problem}")

def extract_stego(messages: list[dict], cover: str, model: LlamaCppModel,
                  codec: LLMTextCodec, cfg: StegoConfig) -> str:
    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    cur    = p_ids.copy()
    cover_bytes = cover.encode("utf-8")

    model.reset(); model.eval(p_ids)
    lg = model.get_logits()
    sp = _make_syncpool(model, cfg)

    dc = Discop(key=cfg.crypto_key, precision=32)
    pos = 0; extracted = []

    while pos < len(cover_bytes):
        pstep = sp.step(lg, cur)
        cdf   = dc.build_cdf(pstep.masses)
        m = SyncPool.match(pstep, cover_bytes, pos)
        if m is None:
            break

        sym_idx, rep_id, nbytes = m
        bits, _ = dc.extract(cdf, cur, sym_idx)
        if bits is None:
            print("\n  [!] tree desync"); break

        extracted.extend(bits)
        pos += nbytes; cur.append(rep_id)
        model.eval([rep_id]); lg = model.get_logits()

    print()

    nonce_bits = getattr(cfg, "nonce_bits", 32)
    if len(extracted) < nonce_bits + CRC_BITS:
        raise ValueError("cover too short — corrupted or desynced; refusing to return garbage")

    # Clear-text nonce, then decrypt the rest (payload + arbitrary tail garbage bits)
    nonce    = extracted[:nonce_bits]
    dec_bits = stream_cipher(extracted[nonce_bits:], cfg.crypto_key, pack_bits(nonce))
    crc_recv = dec_bits[:CRC_BITS]

    print("  LLM decompression …")

    # The AC gracefully processes the stream, hits eos_id, and ignores tail garbage
    secret = codec.decode_bits(dec_bits[CRC_BITS:])

    # Strictly validate against the embedded CRC
    if not check_crc(secret.encode("utf-8"), crc_recv):
        raise ValueError("CRC mismatch — corrupted or desynced; refusing to return garbage")

    return secret
