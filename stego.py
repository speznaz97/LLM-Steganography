import struct
import numpy as np
from typing import Optional

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from arithmetic import ArithmeticCoder
from utils import np_softmax, np_topk, pack_bits, unpack_bits

def _rep_penalty(logits: np.ndarray, ids: list[int], plen: int, penalty: float):
    seen = list(set(ids[plen:]))
    if not seen: return
    ix  = np.array(seen, dtype=np.intp)
    sel = logits[ix]
    logits[ix] = np.where(sel > 0, sel / penalty, sel * penalty)

def _safe_probs(logits: np.ndarray, cur_ids: list[int], model: LlamaCppModel, 
                cfg: StegoConfig, temp_override: Optional[float] = None) -> dict[int, float]:
    temp = temp_override or cfg.stego_temp
    probs = np_softmax(logits, temp)
    top_p, top_i = np_topk(probs, cfg.top_k)

    window = cur_ids[-cfg.retoken_window:]
    prefix_decoded = model.detokenize(window)
    valid: dict[int, float] = {}

    for pv, idx in zip(top_p, top_i):
        pv = float(pv)
        if pv < cfg.prob_threshold: break
        iv = int(idx)
        if iv in model.special_ids: continue
        ts = model.detokenize([iv])
        if any(c in ts for c in cfg.banned_chars): continue
        test = window + [iv]
        if model.tokenize(prefix_decoded + ts, add_bos=False, special=False) == test:
            valid[iv] = pv

    if not valid:
        for _, idx in zip(top_p, top_i):
            iv = int(idx)
            if iv not in model.special_ids:
                valid[iv] = float(probs[iv])
                break
        if not valid:
            valid[int(np.argmax(logits))] = 1.0
    return valid

def get_stego_cdf(logits: np.ndarray, model: LlamaCppModel, cur_ids: list[int], 
                  plen: int, total: int, cfg: StegoConfig):
    _rep_penalty(logits, cur_ids, plen, cfg.rep_penalty)
    vp     = _safe_probs(logits, cur_ids, model, cfg)
    tokens = sorted(vp, key=lambda t: (-vp[t], t))
    probs  = np.array([vp[t] for t in tokens], np.float64)
    cdf    = ArithmeticCoder.build_cdf(probs, total)
    return tokens, cdf

def generate_stego(messages: list[dict], secret: str, model: LlamaCppModel, 
                   codec: LLMTextCodec, cfg: StegoConfig) -> str:
    print(f"  Secret: '{secret}'")
    wire, cs = codec.encode(secret)
    num_bits  = struct.unpack('>I', wire[:4])[0]
    bits_list = unpack_bits(wire[4:], num_bits)
    header    =[int(b) for b in format(len(bits_list) ^ cfg.header_xor, f'0{cfg.header_bits}b')]
    full_bits = header + bits_list
    print(f"  LLM-compressed: {len(bits_list)} bits (header {cfg.header_bits})")

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
    enc_bits, step = 0, 0

    while (enc_bits + enc_pending) < len(full_bits) + 2:
        if step > cfg.max_gen_tokens:
            raise RuntimeError(f"Generation exceeded max_tokens limit ({cfg.max_gen_tokens})")
        
        if step % 10 == 0:
            print(f"\r  [Gen] {min(enc_bits, len(full_bits))}/{len(full_bits)} bits  {step} tok", end="", flush=True)

        tokens, cdf = get_stego_cdf(lg, model, cur, plen, total, cfg)
        sym_idx = ac.find_symbol(dec_lo, dec_hi, val, cdf, total)
        chosen  = tokens[sym_idx]

        dec_lo, dec_hi = ac.narrow(dec_lo, dec_hi, cdf, sym_idx, total)
        dec_lo, dec_hi, val = ac.renorm_dec(dec_lo, dec_hi, val, rb)

        enc_lo, enc_hi = ac.narrow(enc_lo, enc_hi, cdf, sym_idx, total)
        enc_lo, enc_hi, enc_pending, new = ac.renorm_enc(enc_lo, enc_hi, enc_pending)
        enc_bits += len(new)

        cur.append(chosen)
        step += 1
        model.eval([chosen])
        lg = model.get_logits()

    tail = 0
    for ts in range(cfg.tail_max):
        _rep_penalty(lg, cur, plen, cfg.rep_penalty)
        vp = _safe_probs(lg, cur, model, cfg, temp_override=1.0)
        bt = max(vp, key=vp.get)
        if bt in model.special_ids: break

        cur.append(bt); tail += 1
        model.eval([bt])
        lg = model.get_logits()

        if ts >= cfg.tail_min:
            chk = model.detokenize(cur[-3:]).rstrip()
            if chk and chk[-1] in cfg.sentence_enders: break

    print(f"\n  Done: {step} tok ({(len(bits_list) / max(step, 1)):.2f} b/t) + {tail} tail")
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

        tokens, cdf = get_stego_cdf(lg, model, cur, plen, total, cfg)
        actual = r_ids[i] if r_ids[i] in tokens else tokens[0]
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
    wire    = struct.pack('>I', len(payload)) + pack_bits(payload)
    print("  LLM decompression …")
    return codec.decode(wire)