import struct
import numpy as np
import hashlib
from typing import Optional

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from arithmetic import ArithmeticCoder
from utils import np_softmax, pack_bits, unpack_bits, stream_cipher

from syncpool import SyncPool
from discop import Discop

def generate_stego(messages: list[dict], secret: str, model: LlamaCppModel, 
                   codec: LLMTextCodec, cfg: StegoConfig) -> str:
    print(f"  Secret: '{secret}'")
    wire, cs = codec.encode(secret)
    num_bits  = struct.unpack('>I', wire[:4])[0]
    raw_bits = unpack_bits(wire[4:], num_bits)
    # 1. Encrypt bits to ensure they are mathematically uniform
    crypto_key = getattr(cfg, 'crypto_key')
    enc_bits = stream_cipher(raw_bits, crypto_key)
    
    header    =[int(b) for b in format(len(enc_bits) ^ cfg.header_xor, f'0{cfg.header_bits}b')]
    assert (len(enc_bits) ^ cfg.header_xor) < (1 << cfg.header_bits), "payload too long for header_bits"
    full_bits = header + enc_bits
    print(f"  LLM-compressed: {len(enc_bits)} bits (header {cfg.header_bits})")

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
        if step > cfg.max_gen_tokens:
            raise RuntimeError(
                f"exceeded max_tokens ({cfg.max_gen_tokens}); embedded only "
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
    for ts in range(cfg.tail_max):
        recent = model.detokenize(cur[-6:]).rstrip()
        last = recent[-1] if recent else ""
        if last in cfg.sentence_enders:                 # sentence complete -> done
            break
        if tail >= 3 and recent and not any(c.isalnum() for c in recent):
            break                                       # emoji/symbol spam -> bail
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
    plen   = len(p_ids)
    cur    = p_ids.copy()
    cover_bytes = cover.encode("utf-8")          # sender emitted these exact bytes

    model.reset(); model.eval(p_ids)
    lg = model.get_logits()
    ac, total = codec.ac, codec.total
    sp = SyncPool(model, key=cfg.crypto_key,
                temperature=getattr(cfg, "stego_temp", 1.0),
                top_k=getattr(cfg, "syncpool_topk", 256),
                prob_floor=getattr(cfg, "syncpool_floor", 1e-5),
                banned_bytes=tuple(cfg.banned_chars))

    lo, hi, pending = 0, ac.FULL, 0
    dc = Discop(key=cfg.crypto_key, precision=32)
    cover_bytes = cover.encode("utf-8")
    pos = 0; extracted = []; target_len = None

    while pos < len(cover_bytes):
        pstep = sp.step(lg, cur)
        cdf   = dc.build_cdf(pstep.masses)
        m = SyncPool.match(pstep, cover_bytes, pos)
        if m is None:
            print(f"\n  [!] byte desync at {pos}"); break
        sym_idx, rep_id, nbytes = m
        bits, _ = dc.extract(cdf, cur, sym_idx)
        if bits is None:
            print("\n  [!] tree desync"); break             # guard; shouldn't fire
        extracted.extend(bits)

        if target_len is None and len(extracted) >= cfg.header_bits:
            raw = int(''.join(map(str, extracted[:cfg.header_bits])), 2)
            target_len = raw ^ cfg.header_xor
        if target_len is not None and len(extracted) >= cfg.header_bits + target_len:
            extracted = extracted[:cfg.header_bits + target_len]; break

        pos += nbytes; cur.append(rep_id)
        model.eval([rep_id]); lg = model.get_logits()
    print()
    payload = extracted[cfg.header_bits:]
    
    # 4. Decrypt bits before passing them to the decoder
    crypto_key = getattr(cfg, 'crypto_key')
    dec_bits = stream_cipher(payload, crypto_key)
    
    wire    = struct.pack('>I', len(dec_bits)) + pack_bits(dec_bits)
    print("  LLM decompression …")
    return codec.decode(wire)