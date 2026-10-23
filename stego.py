import numpy as np
import hashlib
import json
from typing import Optional

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from arithmetic import ArithmeticCoder
from utils import np_softmax, stream_cipher, unpack_bits

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

def _context_digest(messages: list[dict]) -> bytes:
    """Canonical hash of the conversation -- identical on both endpoints, and
    different for every message since the history grows each turn."""
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(b"|ctx|" + blob.encode("utf-8")).digest()

def _attempt_mask(key, ctx: bytes, nbits: int) -> list[int]:
    """Keyed pad for the clear attempt counter, so its low-entropy bits (0, 1,
    2, ...) still look uniform to Discop and to anyone without the key."""
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    pad = hashlib.sha256(key_bytes + b"|attempt|" + ctx).digest()
    return unpack_bits(pad, nbits)

def _derive_nonce(ctx: bytes, attempt: int) -> bytes:
    """Per-message keystream nonce from the conversation + attempt counter.
    Caveat: re-sending in an identical context with the same counter reuses
    the keystream."""
    return hashlib.sha256(b"|nonce|" + ctx + attempt.to_bytes(4, "big")).digest()[:16]

def _end_ids(model: LlamaCppModel) -> tuple[int, ...]:
    """Tokens that end the assistant turn: eos plus <|im_end|> (the same id on Qwen)."""
    ids = {model.eos_id} if model.eos_id is not None and model.eos_id >= 0 else set()
    im_end = model.tokenize("<|im_end|>", add_bos=False, special=True)
    if len(im_end) == 1:
        ids.add(im_end[0])
    return tuple(sorted(ids))

_SPLIT_MARK = "\x00stego-split\x00"

def _followup_ids(messages: list[dict], model: LlamaCppModel) -> list[int]:
    """Tokens that close one assistant message and open the next assistant turn.

    Taken from the chat template itself (render history + a marker assistant
    message + generation prompt, keep what follows the marker), then appended
    token-wise after the message's emitted tokens -- re-rendering the history
    would re-tokenize the cover canonically, which SyncPool's tokens are not.
    Both endpoints derive it from the same original `messages`."""
    probe = messages + [{"role": "assistant", "content": _SPLIT_MARK}]
    text = model.apply_chat_template(probe, add_generation_prompt=True, enable_thinking=False)
    i = text.rfind(_SPLIT_MARK)
    seam = (text[i + len(_SPLIT_MARK):] if i >= 0
            else "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return model.tokenize(seam, add_bos=False, special=True)

def _end_allowed(msg_idx: int, msg_step: int, cfg: StegoConfig) -> bool:
    """End-of-turn is an ordinary payload symbol, except as a message's first
    token (an empty message can't be sent) and in the last allowed message
    (the cap). Applied identically by sender and receiver."""
    return msg_step > 0 and msg_idx < max(1, getattr(cfg, "max_messages", 3)) - 1

def _make_syncpool(model: LlamaCppModel, cfg: StegoConfig) -> SyncPool:
    return SyncPool(model, key=cfg.crypto_key,
                    temperature=getattr(cfg, "stego_temp", 1.0),
                    top_k=getattr(cfg, "syncpool_topk", 256),
                    prob_floor=getattr(cfg, "syncpool_floor", 1e-5),
                    banned_bytes=tuple(cfg.banned_chars),
                    end_ids=_end_ids(model),
                    min_p=getattr(cfg, "syncpool_minp", 0.0))

def _cover_problem(cover: str, emitted: list[int], sp: SyncPool,
                   cfg: StegoConfig) -> Optional[str]:
    """Why the receiver could NOT re-derive this cover from its text, or None.

    The receiver only ever sees `cover.encode("utf-8")` (after whatever the
    transport does to it) and must match it byte-for-byte against SyncPool
    representatives, so the sender checks that before sending anything."""
    if not cover:
        return "empty message"
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
    """One generation pass. Returns (covers, problem-or-None, payload_tok, tail_tok)."""
    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    cur    = p_ids.copy()
    seam   = _followup_ids(messages, model)

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

    spans = []                  # (start, end) of each finished message in cur
    start = len(cur)
    msg_step = 0
    ended = False               # last message already closed by an end-of-turn choice
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

        pstep = sp.step(lg, cur, allow_end=_end_allowed(len(spans), msg_step, cfg))
        cdf   = dc.build_cdf(pstep.masses)
        sym_idx, nb = dc.embed(cdf, cur, read_bit)
        step += 1

        if sym_idx == pstep.end_idx:
            # The model chose to stop: close this message and, if bits remain,
            # carry on in a follow-up assistant turn. The end token itself is
            # replaced by the template's seam on both sides
            spans.append((start, len(cur)))
            if pos >= len(full_bits):
                ended = True
                break
            cur.extend(seam)
            model.eval(seam)
            lg = model.get_logits()
            start = len(cur)
            msg_step = 0
            continue

        chosen = pstep.reps[sym_idx]
        cur.append(chosen)
        msg_step += 1
        model.eval([chosen])
        lg = model.get_logits()

    tail_bit_reader = _make_tail_bit_generator(cfg.crypto_key, cur)

    tail_step = 0
    tail_min = getattr(cfg, "tail_min", 0)
    tail_max = getattr(cfg, "tail_max", 30)
    sentence_enders = getattr(cfg, "sentence_enders", ['.', '!', '?', '\n'])

    while not ended and tail_step < tail_max:
        # Stop as soon as the text ends a sentence -- including right after the
        # payload (tail_min=0), so a cleanly-ended payload gets no tail at all
        if tail_step >= tail_min:
            recent = model.detokenize(cur[max(start, len(cur) - 6):]).rstrip()
            last = recent[-1] if recent else ""
            if last in sentence_enders:
                break
            if recent and not any(c.isalnum() for c in recent):
                break

        # Step SyncPool and Discop using the exact same parameters as the payload.
        # End-of-turn is a candidate here even in the capped message: the
        # receiver never needs tail bits, so it needn't mirror this
        pstep = sp.step(lg, cur, allow_end=msg_step > 0)
        cdf   = dc.build_cdf(pstep.masses)

        # Embed secure pseudo-random bits to preserve distribution
        sym_idx, nb = dc.embed(cdf, cur, tail_bit_reader)
        tail_step += 1
        if sym_idx == pstep.end_idx:
            break                                   # the model ended its turn
        chosen = pstep.reps[sym_idx]

        cur.append(chosen)
        msg_step += 1
        model.eval([chosen])
        lg = model.get_logits()

    if not ended:
        spans.append((start, len(cur)))

    covers = [model.detokenize(cur[a:b], skip_special=True) for a, b in spans]
    problem = None
    for i, ((a, b), cover) in enumerate(zip(spans, covers)):
        problem = _cover_problem(cover, cur[a:b], sp, cfg)
        if problem is not None:
            problem = f"message {i + 1}/{len(covers)}: {problem}"
            break
    return covers, problem, step, tail_step, (cur, spans)

def generate_stego(messages: list[dict], secret: str, model: LlamaCppModel,
                   codec: LLMTextCodec, cfg: StegoConfig,
                   stats: Optional[dict] = None) -> list[str]:
    """Returns the cover as one or more consecutive assistant messages, to be
    sent in order: the model may end its turn mid-payload, and the rest of the
    payload follows in a new turn. One keystream, keyed by `messages`, spans them all.

    If `stats` is given it is filled in for benchmarking: the exact token ids
    sent (`ids`: prompt + messages + seams, `spans`: each message's slice of
    it), payload/tail token counts, bit counts, the attempt that succeeded and
    whether the message cap was hit."""
    print(f"  Secret: '{secret}'")

    # Zero-header payload with CRC prefix
    raw_bits, _ = codec.encode_bits(secret)
    payload   = crc_bits(secret.encode("utf-8")) + raw_bits
    attempt_bits = getattr(cfg, "attempt_bits", 3)
    print(f"  payload: {len(raw_bits)} body + {CRC_BITS} CRC + {attempt_bits} attempt bits, "
          f"0 length-header bits")

    ctx  = _context_digest(messages)
    mask = _attempt_mask(cfg.crypto_key, ctx, attempt_bits)
    attempts = max(1, min(getattr(cfg, "gen_attempts", 3), 2 ** attempt_bits))
    for attempt in range(1, attempts + 1):
        # Nonce derived from the conversation; only the (masked) attempt counter
        # goes in the clear ahead of the ciphertext, so every retry keys a
        # different keystream and so a different cover
        counter = attempt - 1
        clear = [b ^ m for b, m in zip(unpack_bits(counter.to_bytes(4, "big"), 32)[-attempt_bits:], mask)]
        full_bits = clear + stream_cipher(payload, cfg.crypto_key, _derive_nonce(ctx, counter))

        covers, problem, step, tail_step, (ids, spans) = _embed(messages, full_bits, model, cfg)
        print(f"\n  Done: {step} tok ({(len(raw_bits) / max(step, 1)):.2f} b/t) + {tail_step} tail tokens, "
              f"{len(covers)} message(s)")
        if problem is None:
            if stats is not None:
                # capped: the cover used every allowed message, so the last one
                # could not end early (end-of-turn is disallowed there)
                stats.update(ids=ids, spans=spans, payload_tokens=step, tail_tokens=tail_step,
                             body_bits=len(raw_bits), total_bits=len(full_bits), attempt=attempt,
                             capped=len(covers) >= max(1, getattr(cfg, "max_messages", 3)))
            return covers
        print(f"  [!] attempt {attempt}/{attempts} rejected: {problem}")

    raise RuntimeError(f"no transmissible cover after {attempts} attempts: {problem}")

def extract_stego(messages: list[dict], covers: list[str], model: LlamaCppModel,
                  codec: LLMTextCodec, cfg: StegoConfig) -> str:
    """`messages` is the context BEFORE the first cover message (the same one
    the sender used); `covers` are generate_stego's messages, in order."""
    if isinstance(covers, str):
        covers = [covers]
    max_msgs = max(1, getattr(cfg, "max_messages", 3))
    if not covers or len(covers) > max_msgs:
        raise ValueError(f"expected 1..{max_msgs} cover messages, got {len(covers)}")

    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    cur    = p_ids.copy()
    seam   = _followup_ids(messages, model)

    model.reset(); model.eval(p_ids)
    lg = model.get_logits()
    sp = _make_syncpool(model, cfg)

    dc = Discop(key=cfg.crypto_key, precision=32)
    extracted = []; desync = False

    for i, cover in enumerate(covers):
        if i > 0:
            cur.extend(seam)
            model.eval(seam); lg = model.get_logits()
        cover_bytes = cover.encode("utf-8")
        pos = 0; msg_step = 0

        while True:
            pstep = sp.step(lg, cur, allow_end=_end_allowed(i, msg_step, cfg))
            cdf   = dc.build_cdf(pstep.masses)

            if pos >= len(cover_bytes):
                # Text over: the sender chose end-of-turn here, so recover the
                # bits that choice carried. On the last message these are tail
                # garbage (or the final payload bits, if the payload ran out
                # exactly on that choice) -- the AC ignores anything extra
                if pstep.end_idx >= 0:
                    bits, _ = dc.extract(cdf, cur, pstep.end_idx)
                    extracted.extend(bits or [])
                elif i < len(covers) - 1:
                    print(f"\n  [!] message {i + 1} ended where end-of-turn was impossible")
                    desync = True
                break

            m = SyncPool.match(pstep, cover_bytes, pos)
            if m is None:
                desync = True; break

            sym_idx, rep_id, nbytes = m
            bits, _ = dc.extract(cdf, cur, sym_idx)
            if bits is None:
                print("\n  [!] tree desync"); desync = True; break

            extracted.extend(bits)
            pos += nbytes; cur.append(rep_id); msg_step += 1
            model.eval([rep_id]); lg = model.get_logits()

        if desync:
            break

    print()

    attempt_bits = getattr(cfg, "attempt_bits", 3)
    if len(extracted) < attempt_bits + CRC_BITS:
        raise ValueError("cover too short — corrupted or desynced; refusing to return garbage")

    # Masked attempt counter, then re-derive the nonce from the same conversation
    # and decrypt the rest (payload + arbitrary tail garbage bits)
    ctx      = _context_digest(messages)
    mask     = _attempt_mask(cfg.crypto_key, ctx, attempt_bits)
    clear    = [b ^ m for b, m in zip(extracted[:attempt_bits], mask)]
    counter  = int("".join(map(str, clear)), 2) if clear else 0
    dec_bits = stream_cipher(extracted[attempt_bits:], cfg.crypto_key, _derive_nonce(ctx, counter))
    crc_recv = dec_bits[:CRC_BITS]

    print("  LLM decompression …")

    # The AC gracefully processes the stream, hits eos_id, and ignores tail garbage
    secret = codec.decode_bits(dec_bits[CRC_BITS:])

    # Strictly validate against the embedded CRC
    if not check_crc(secret.encode("utf-8"), crc_recv):
        raise ValueError("CRC mismatch — corrupted or desynced; refusing to return garbage")

    return secret
