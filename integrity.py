"""
Integrity & verification for the stego channel.

Two independent guards:

1. Payload CRC  -- a fixed 16-bit checksum over the *secret*, embedded as the
   first bits of the payload (before the AC-coded body). After extraction the
   receiver recomputes it over the decoded secret; a mismatch means the payload
   was corrupted or the channel desynced, so we refuse to return garbage. This
   is what turns a silent wrong-message (e.g. a dropped character) into a loud,
   detected failure. It pairs with the zero-header design: the CRC is a fixed
   prefix, so the receiver can always read it without a length header, and the
   AC body after it self-terminates at eos.

2. Connect-time handshake -- both endpoints run identical probe prompts through
   *both* models (cover and codec) and hash the resulting integer CDFs. If the
   fingerprints differ, the two sides do not compute identical distributions
   (different model file, quantization, backend, flash-attention, or llm.py
   version) and the channel WILL desync. Refuse to start. This catches the most
   common and most invisible failure mode before any message is sent.

The fingerprint is deliberately key-INDEPENDENT: it hashes pool *masses* (the
distribution), never representatives (which depend on the crypto key), so it can
be exchanged in the clear without touching the shared secret.
"""

from __future__ import annotations
import binascii
import hashlib

import numpy as np

# ----------------------------------------------------------------------------
# Payload CRC
# ----------------------------------------------------------------------------
CRC_BITS = 16


def crc_bits(data: bytes) -> list[int]:
    """16-bit CRC of `data` as an MSB-first bit list."""
    # Use CRC-16-CCITT or similar, or simply mask the standard CRC32 to 16 bits
    c = binascii.crc32(data) & 0xFFFF  # Mask to 16-bit
    return [(c >> (CRC_BITS - 1 - i)) & 1 for i in range(CRC_BITS)]


def check_crc(data: bytes, bits: list[int]) -> bool:
    if len(bits) < CRC_BITS:
        return False
    got = 0
    for b in bits[:CRC_BITS]:
        got = (got << 1) | (b & 1)
    return got == (binascii.crc32(data) & 0xFFFF)


# ----------------------------------------------------------------------------
# Connect-time handshake (distribution fingerprint)
# ----------------------------------------------------------------------------
_PROBE_MESSAGES = [
    [{"role": "system", "content": "You are a helpful assistant. Reply naturally."},
     {"role": "user", "content": "Tell me a little about how your week has been going."}],
    [{"role": "system", "content": "Answer concisely."},
     {"role": "user", "content": "What is the capital of France, and one fact about it?"}],
]
_PROBE_STEPS = 12


def _next_advance_token(lg: np.ndarray, special_ids) -> int:
    """Deterministic, key-INDEPENDENT trajectory: most probable non-special token."""
    for t in np.argsort(-lg):
        it = int(t)
        if it not in special_ids:
            return it
    return int(np.argmax(lg))


def _cover_digest(model, cfg) -> str:
    from syncpool import SyncPool
    from discop import Discop
    # key here is irrelevant: we hash masses, which do not depend on it
    sp = SyncPool(model, key="probe",
                  temperature=getattr(cfg, "stego_temp", 1.0),
                  top_k=getattr(cfg, "syncpool_topk", 256),
                  prob_floor=getattr(cfg, "syncpool_floor", 1e-5),
                  banned_bytes=tuple(cfg.banned_chars))
    dc = Discop(key="probe", precision=32)
    h = hashlib.sha256()
    for msgs in _PROBE_MESSAGES:
        prompt = model.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        ids = model.tokenize(prompt, add_bos=False, special=True)
        model.reset(); model.eval(ids)
        lg = model.get_logits()
        cur = list(ids)
        for _ in range(_PROBE_STEPS):
            ps = sp.step(lg, cur)
            h.update(dc.build_cdf(ps.masses).tobytes())     # integer CDF == distribution
            tok = _next_advance_token(lg, model.special_ids)
            cur.append(tok); model.eval([tok])
            lg = model.get_logits()
    return h.hexdigest()


def _codec_digest(codec) -> str:
    m = codec.model
    m.reset(); m.eval(codec.primer_ids)
    lg = m.get_logits()
    h = hashlib.sha256()
    for _ in range(_PROBE_STEPS):
        h.update(np.asarray(codec._logits_to_cdf(lg)).tobytes())
        tok = _next_advance_token(lg, m.special_ids)
        m.eval([tok])
        lg = m.get_logits()
    return h.hexdigest()


def session_fingerprint(model, codec, cfg) -> str:
    """A short hex fingerprint of everything that must match between endpoints.
    Exchange it before the first message and compare via require_match()."""
    wire_params = (
        f"crc={CRC_BITS}|precision=32|"
        f"stego_temp={getattr(cfg, 'stego_temp', 1.0)}|"
        f"topk={getattr(cfg, 'syncpool_topk', 256)}|"
        f"floor={getattr(cfg, 'syncpool_floor', 1e-5)}|"
        f"banned={tuple(cfg.banned_chars)}"
    )
    blob = "||".join([_cover_digest(model, cfg), _codec_digest(codec), wire_params])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def require_match(local_fp: str, remote_fp: str) -> None:
    """Raise if the two endpoints are not provably compatible."""
    if local_fp != remote_fp:
        raise RuntimeError(
            "HANDSHAKE FAILED — endpoints compute different distributions and "
            "will desync.\n"
            f"  local : {local_fp}\n"
            f"  remote: {remote_fp}\n"
            "Causes: different model file, quantization, GPU/CPU backend, "
            "flash-attention, or llm.py version. Do not send messages."
        )
    print(f"  [handshake] OK — endpoints match ({local_fp})")