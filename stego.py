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

def _rep_penalty(logits: np.ndarray, ids: list[int], plen: int, penalty: float):
    seen = list(set(ids[plen:]))
    if not seen: return
    ix  = np.array(seen, dtype=np.intp)
    sel = logits[ix]
    logits[ix] = np.where(sel > 0, sel / penalty, sel * penalty)

def _safe_probs(logits: np.ndarray, cur_ids: list[int], plen: int, prompt: str, 
                model: LlamaCppModel, cfg: StegoConfig, temp_override: Optional[float] = None) -> dict[int, float]:
    temp = temp_override or getattr(cfg, 'stego_temp', 1.0)
    probs = np_softmax(logits, temp)
    order = np.argsort(-probs)
    
    cover_so_far = model.detokenize(cur_ids[plen:], skip_special=True)
    full_text = prompt + cover_so_far
    
    # Fast BPE desync check using suffix: isolates the right-most boundary
    suffix_text = full_text[-64:] if len(full_text) > 64 else full_text
    prefix_ids = model.tokenize(suffix_text, add_bos=False, special=True)
    
    valid: dict[int, float] = {}
    
    for idx in order:
        pv = float(probs[idx])
        if pv < 1e-5: break # Performance cutoff to prevent CPU hangs
        
        iv = int(idx)
        if iv in model.special_ids: continue
        
        ts = model.detokenize([iv])
        if any(c in ts for c in cfg.banned_chars): continue
        
        # Test just the boundary; extremely fast O(1) check
        test_r_ids = model.tokenize(suffix_text + ts, add_bos=False, special=True)
        if test_r_ids == prefix_ids + [iv]:
            valid[iv] = pv
            
    # Fallback to prevent crashes if the entire top probability mass is BPE-unsafe
    if not valid:
        for idx in order:
            iv = int(idx)
            if iv in model.special_ids: continue
            ts = model.detokenize([iv])
            test_r_ids = model.tokenize(suffix_text + ts, add_bos=False, special=True)
            if test_r_ids == prefix_ids + [iv]:
                valid[iv] = float(probs[idx])
                break
        if not valid:
            valid[int(order[0])] = 1.0
            
    return valid

def get_stego_cdf(logits: np.ndarray, model: LlamaCppModel, cur_ids: list[int], 
                  plen: int, prompt: str, total: int, cfg: StegoConfig):
    _rep_penalty(logits, cur_ids, plen, cfg.rep_penalty)
    vp = _safe_probs(logits, cur_ids, plen, prompt, model, cfg)
    rounded_vp = {t: round(p, 5) for t, p in vp.items()}
    tokens = sorted(rounded_vp, key=lambda t: (-rounded_vp[t], t))
    probs = np.array([rounded_vp[t] for t in tokens], np.float64)
    cdf = ArithmeticCoder.build_cdf(probs, total)
    return tokens, cdf

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
    full_bits = header + enc_bits
    print(f"  LLM-compressed: {len(enc_bits)} bits (header {cfg.header_bits})")

    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    plen   = len(p_ids)
    cur    = p_ids.copy()

    model.reset()

    sp = SyncPool(
        model,
        key=cfg.crypto_key,                       # share the SAME key both sides
        temperature=getattr(cfg, "stego_temp", 1.0),
        top_k=getattr(cfg, "syncpool_topk", 256),
        prob_floor=getattr(cfg, "syncpool_floor", 1e-5),
        banned_bytes=tuple(cfg.banned_chars),
    )

    model.eval(p_ids)
    lg = model.get_logits()

    ac, total = codec.ac, codec.total
    pos = 0

    def rb():
        nonlocal pos
        b = full_bits[pos] if pos < len(full_bits) else (1 if pos == len(full_bits) else 0)
        pos += 1
        return b

    dec_lo, dec_hi, val = 0, ac.FULL, 0
    for _ in range(ac.P): val = (val << 1) | rb()

    enc_lo, enc_hi, enc_pending = 0, ac.FULL, 0
    enc_bits_count, step = 0, 0

    while (enc_bits_count + enc_pending) < len(full_bits) + 2:
        if step > cfg.max_gen_tokens:
            raise RuntimeError(f"Generation exceeded max_tokens limit ({cfg.max_gen_tokens})")
        
        if step % 10 == 0:
            print(f"\r  [Gen] {min(enc_bits_count, len(full_bits))}/{len(full_bits)} bits  {step} tok", end="", flush=True)

        _rep_penalty(lg, cur, plen, cfg.rep_penalty)        # keep iff also kept in extract
        pstep   = sp.step(lg, cur)
        cdf     = ArithmeticCoder.build_cdf(pstep.masses, total)
        sym_idx = ac.find_symbol(dec_lo, dec_hi, val, cdf, total)
        chosen  = pstep.reps[sym_idx]
        #tokens, cdf = get_stego_cdf(lg, model, cur, plen, prompt, total, cfg)
        #sym_idx = ac.find_symbol(dec_lo, dec_hi, val, cdf, total)
        #chosen  = tokens[sym_idx]

        dec_lo, dec_hi = ac.narrow(dec_lo, dec_hi, cdf, sym_idx, total)
        dec_lo, dec_hi, val = ac.renorm_dec(dec_lo, dec_hi, val, rb)

        enc_lo, enc_hi = ac.narrow(enc_lo, enc_hi, cdf, sym_idx, total)
        enc_lo, enc_hi, enc_pending, new = ac.renorm_enc(enc_lo, enc_hi, enc_pending)
        enc_bits_count += len(new)

        cur.append(chosen)
        step += 1
        model.eval([chosen])
        lg = model.get_logits()

    tail = 0
    for ts in range(cfg.tail_max):
        # Evaluate the model's natural top choice
        probs_natural = np_softmax(lg, 1.0)
        natural_top = int(np.argmax(probs_natural))
        
        # Check if the text currently ends with punctuation
        chk_current = model.detokenize(cur[-3:]).rstrip()
        ends_with_punctuation = bool(chk_current and chk_current[-1] in cfg.sentence_enders)
        
        # Only allow a natural stop if the sentence is complete AND we have generated enough tail
        if natural_top in model.special_ids:
            if ends_with_punctuation and ts >= cfg.tail_min: # Changed 'or' to 'and'
                break
                
        _rep_penalty(lg, cur, plen, cfg.rep_penalty)
        vp = _safe_probs(lg, cur, plen, prompt, model, cfg, temp_override=1.0)
        bt = max(vp, key=vp.get)
        if bt in model.special_ids: break
        
        cur.append(bt); tail += 1
        model.eval([bt])
        lg = model.get_logits()
        
        if ts >= cfg.tail_min:
            chk = model.detokenize(cur[-3:]).rstrip()
            if chk and chk[-1] in cfg.sentence_enders: break

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
    extracted, target_len = [], None
    pos = 0

    while pos < len(cover_bytes):
        _rep_penalty(lg, cur, plen, cfg.rep_penalty)     # match generate exactly
        pstep = sp.step(lg, cur)
        cdf   = ArithmeticCoder.build_cdf(pstep.masses, total)

        m = SyncPool.match(pstep, cover_bytes, pos)
        if m is None:
            print(f"\n  [!] desync: no pool matches cover at byte {pos}")
            break
        sym_idx, rep_id, nbytes = m

        lo, hi = ac.narrow(lo, hi, cdf, sym_idx, total)
        while True:
            if hi < ac.HALF:
                extracted.append(0); extracted.extend([1] * pending); pending = 0
            elif lo >= ac.HALF:
                extracted.append(1); extracted.extend([0] * pending); pending = 0
                lo -= ac.HALF; hi -= ac.HALF
            elif lo >= ac.QTR and hi < 3 * ac.QTR:
                pending += 1; lo -= ac.QTR; hi -= ac.QTR
            else:
                break
            lo <<= 1; hi = (hi << 1) | 1

        if target_len is None and len(extracted) >= cfg.header_bits:
            raw = int(''.join(map(str, extracted[:cfg.header_bits])), 2)
            target_len = raw ^ cfg.header_xor
        if target_len is not None and len(extracted) >= cfg.header_bits + target_len:
            extracted = extracted[:cfg.header_bits + target_len]
            break

        pos += nbytes
        cur.append(rep_id)
        model.eval([rep_id])
        lg = model.get_logits()
    print()
    payload = extracted[cfg.header_bits:]
    
    # 4. Decrypt bits before passing them to the decoder
    crypto_key = getattr(cfg, 'crypto_key')
    dec_bits = stream_cipher(payload, crypto_key)
    
    wire    = struct.pack('>I', len(dec_bits)) + pack_bits(dec_bits)
    print("  LLM decompression …")
    return codec.decode(wire)