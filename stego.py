import numpy as np
import hashlib
from typing import Optional

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from arithmetic import ArithmeticCoder
from utils import np_softmax, stream_cipher

from syncpool import SyncPool
from discop import Discop
from integrity import crc_bits, check_crc, CRC_BITS

def generate_stego(messages: list[dict], secret: str, model: LlamaCppModel, 
                   codec: LLMTextCodec, cfg: StegoConfig) -> str:
    print(f"  Secret: '{secret}'")
    
    # Zero-header payload with CRC prefix
    raw_bits, _ = codec.encode_bits(secret)
    payload   = crc_bits(secret.encode("utf-8")) + raw_bits
    full_bits = stream_cipher(payload, cfg.crypto_key)
    print(f"  payload: {len(raw_bits)} body + {CRC_BITS} CRC bits, 0 length-header bits")

    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    plen   = len(p_ids)
    cur    = p_ids.copy()

    model.reset()

    sp = SyncPool(model, key=cfg.crypto_key,
                  temperature=getattr(cfg, "stego_temp", 1.0),
                  top_k=getattr(cfg, "syncpool_topk", 256),
                  prob_floor=getattr(cfg, "syncpool_floor", 1e-5),
                  banned_bytes=tuple(cfg.banned_chars))
    
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

    tail = 0
    for ts in range(getattr(cfg, "tail_max", 20)):
        recent = model.detokenize(cur[-6:]).rstrip()
        last = recent[-1] if recent else ""
        if last in getattr(cfg, "sentence_enders", ['.', '!', '?']):
            break
        if tail >= 3 and recent and not any(c.isalnum() for c in recent):
            break
        probs = np_softmax(lg, 1.0)
        order = np.argsort(-probs)
        nt = next((int(i) for i in order if int(i) not in model.special_ids), None)
        if nt is None:
            break
        cur.append(nt); tail += 1
        model.eval([nt]); lg = model.get_logits()

    print(f"\n  Done: {step} tok ({(len(raw_bits) / max(step, 1)):.2f} b/t) + {tail} tail")
    return model.detokenize(cur[plen:], skip_special=True)

def extract_stego(messages: list[dict], cover: str, model: LlamaCppModel, 
                  codec: LLMTextCodec, cfg: StegoConfig) -> str:
    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    cur    = p_ids.copy()
    cover_bytes = cover.encode("utf-8")

    model.reset(); model.eval(p_ids)
    lg = model.get_logits()
    sp = SyncPool(model, key=cfg.crypto_key,
                temperature=getattr(cfg, "stego_temp", 1.0),
                top_k=getattr(cfg, "syncpool_topk", 256),
                prob_floor=getattr(cfg, "syncpool_floor", 1e-5),
                banned_bytes=tuple(cfg.banned_chars))

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
    
    # Decrypt all extracted bits (payload + arbitrary tail garbage bits)
    dec_bits = stream_cipher(extracted, cfg.crypto_key)
    crc_recv = dec_bits[:CRC_BITS]
    
    print("  LLM decompression …")
    
    # The AC gracefully processes the stream, hits eos_id, and ignores tail garbage
    secret = codec.decode_bits(dec_bits[CRC_BITS:])
    
    # Strictly validate against the embedded CRC
    if not check_crc(secret.encode("utf-8"), crc_recv):
        raise ValueError("CRC mismatch — corrupted or desynced; refusing to return garbage")
        
    return secret