import struct
import numpy as np
import hashlib
from typing import Optional

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from arithmetic import ArithmeticCoder
from utils import np_softmax, pack_bits, unpack_bits

def stream_cipher(bits: list[int], key: str) -> list[int]:
    """Lightweight CTR stream cipher to guarantee uniformly random payload bits."""
    out_bits = []
    counter = 0
    while len(out_bits) < len(bits):
        block = hashlib.sha256(f"{key}_{counter}".encode('utf-8')).digest()
        for byte in block:
            for i in range(8):
                out_bits.append((byte >> (7 - i)) & 1)
        counter += 1
    return [b ^ k for b, k in zip(bits, out_bits[:len(bits)])]

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
    
    # Simulate EXACTLY what the receiver will see when they concatenate prompt + cover
    cover_so_far = model.detokenize(cur_ids[plen:], skip_special=True)
    full_text = prompt + cover_so_far
    
    valid: dict[int, float] = {}
    
    for idx in order:
        pv = float(probs[idx])
        if pv < 1e-5: break # Performance cutoff to prevent CPU hangs
        
        iv = int(idx)
        if iv in model.special_ids: continue
        
        ts = model.detokenize([iv])
        if any(c in ts for c in cfg.banned_chars): continue
        
        # 100% Bulletproof BPE Desync check:
        # Tokenize the complete text with the new candidate token appended.
        # It MUST exactly equal our current token sequence, otherwise the receiver will desync!
        test_r_ids = model.tokenize(full_text + ts, add_bos=False, special=True)
        if test_r_ids == cur_ids + [iv]:
            valid[iv] = pv
            
    # Fallback to prevent crashes if the entire top probability mass is BPE-unsafe
    if not valid:
        for idx in order:
            iv = int(idx)
            if iv in model.special_ids: continue
            ts = model.detokenize([iv])
            test_r_ids = model.tokenize(full_text + ts, add_bos=False, special=True)
            if test_r_ids == cur_ids + [iv]:
                valid[iv] = float(probs[idx])
                break
        if not valid:
            valid[int(order[0])] = 1.0
            
    return valid

def get_stego_cdf(logits: np.ndarray, model: LlamaCppModel, cur_ids: list[int], 
                  plen: int, prompt: str, total: int, cfg: StegoConfig):
    _rep_penalty(logits, cur_ids, plen, cfg.rep_penalty)
    vp = _safe_probs(logits, cur_ids, plen, prompt, model, cfg)
    tokens = sorted(vp, key=lambda t: (-vp[t], t))
    probs = np.array([vp[t] for t in tokens], np.float64)
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

        tokens, cdf = get_stego_cdf(lg, model, cur, plen, prompt, total, cfg)
        sym_idx = ac.find_symbol(dec_lo, dec_hi, val, cdf, total)
        chosen  = tokens[sym_idx]

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
    full   = prompt + cover
    r_ids  = model.tokenize(full, add_bos=False, special=True)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    plen   = len(p_ids)
    cur    = p_ids.copy()

    model.reset()
    model.eval(p_ids)
    lg = model.get_logits()

    ac, total = codec.ac, codec.total
    lo, hi, pending = 0, ac.FULL, 0
    extracted, target_len =[], None

    for i in range(plen, len(r_ids)):
        if (i - plen) % 10 == 0:
            print(f"\r  [Ext] {i - plen}/{len(r_ids) - plen} tok  {len(extracted)} bits", end="", flush=True)

        tokens, cdf = get_stego_cdf(lg, model, cur, plen, prompt, total, cfg)
        actual = r_ids[i] if r_ids[i] in tokens else tokens[0]
        
        # 3. Explicitly warn if a desync happens so it doesn't fail silently
        if r_ids[i] not in tokens:
            print(f"\n  [!] Desync warning: token '{model.detokenize([r_ids[i]])}' ({r_ids[i]}) not in valid set!")

        sym_idx = tokens.index(actual)

        lo, hi = ac.narrow(lo, hi, cdf, sym_idx, total)
        while True:
            if hi < ac.HALF:
                extracted.append(0); extracted.extend([1] * pending); pending = 0
            elif lo >= ac.HALF:
                extracted.append(1); extracted.extend([0] * pending); pending = 0
                lo -= ac.HALF; hi -= ac.HALF
            elif lo >= ac.QTR and hi < 3 * ac.QTR:
                pending += 1; lo -= ac.QTR; hi -= ac.QTR
            else: break
            lo <<= 1; hi = (hi << 1) | 1

        if target_len is None and len(extracted) >= cfg.header_bits:
            raw = int(''.join(map(str, extracted[:cfg.header_bits])), 2)
            target_len = raw ^ cfg.header_xor

        if target_len is not None and len(extracted) >= cfg.header_bits + target_len:
            extracted = extracted[:cfg.header_bits + target_len]
            break

        cur.append(r_ids[i])
        model.eval([r_ids[i]])
        lg = model.get_logits()

    print()
    payload = extracted[cfg.header_bits:]
    
    # 4. Decrypt bits before passing them to the decoder
    crypto_key = getattr(cfg, 'crypto_key', 'shared_secret_password_123')
    dec_bits = stream_cipher(payload, crypto_key)
    
    wire    = struct.pack('>I', len(dec_bits)) + pack_bits(dec_bits)
    print("  LLM decompression …")
    return codec.decode(wire)