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


class PayloadTooLong(RuntimeError):
    """The whole payload did not fit in one message: the model ended its turn
    with bits still due. One-shot cannot send this secret; use StegoSender /
    StegoReceiver and spread it across a real back-and-forth."""


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
    """Per-transmission keystream nonce from the conversation at the first cover
    + attempt counter. Caveat: re-sending in an identical context with the same
    counter reuses the keystream."""
    return hashlib.sha256(b"|nonce|" + ctx + attempt.to_bytes(4, "big")).digest()[:16]

def _end_ids(model: LlamaCppModel) -> tuple[int, ...]:
    """Tokens that end the assistant turn: eos plus <|im_end|> (the same id on Qwen)."""
    ids = {model.eos_id} if model.eos_id is not None and model.eos_id >= 0 else set()
    im_end = model.tokenize("<|im_end|>", add_bos=False, special=True)
    if len(im_end) == 1:
        ids.add(im_end[0])
    return tuple(sorted(ids))

def _end_allowed(msg_step: int) -> bool:
    """End-of-turn is an ordinary payload symbol, except as a message's first
    token (an empty message can't be sent). There is no message cap: one turn
    always produces exactly one message, ending where the model chooses.
    Applied identically by sender and receiver."""
    return msg_step > 0

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


def _embed_turn(messages: list[dict], full_bits: list[int], start: int,
                model: LlamaCppModel, cfg: StegoConfig):
    """One assistant turn: embed full_bits[start:] into a single message, ending
    where the model chooses.

    Returns (cover, problem, payload_tok, tail_tok, consumed, complete, ids, span):
      consumed  -- bits of full_bits[start:] embedded this turn
      complete  -- True iff the whole remainder fit (payload exhausted, so the
                   message ran to a natural end and got a sentence-completing
                   tail); False iff the model ended its turn with bits still due,
                   which the caller carries into the next real turn.
    The keystream position is continuous across turns; only `start` moves.
    """
    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    cur    = p_ids.copy()

    sp = _make_syncpool(model, cfg)
    dc = Discop(key=cfg.crypto_key, precision=32)

    model.reset()
    model.eval(p_ids)
    lg = model.get_logits()

    pos = start
    def read_bit():
        nonlocal pos
        b = full_bits[pos] if pos < len(full_bits) else 0
        pos += 1
        return b

    cover_start = len(cur)
    msg_step = 0
    ended_early = False         # the model chose end-of-turn before bits ran out
    step = 0
    while pos < len(full_bits):
        if step > getattr(cfg, "max_gen_tokens", 500):
            raise RuntimeError(
                f"exceeded max_tokens; embedded only "
                f"{pos - start}/{len(full_bits) - start} bits — distribution too "
                f"peaked, raise stego_temp / max_gen_tokens")
        if step % 10 == 0:
            print(f"\r  [Gen] {min(pos, len(full_bits)) - start}/{len(full_bits) - start} bits  {step} tok",
                  end="", flush=True)

        pstep = sp.step(lg, cur, allow_end=_end_allowed(msg_step))
        cdf   = dc.build_cdf(pstep.masses)
        sym_idx, nb = dc.embed(cdf, cur, read_bit)
        step += 1

        if sym_idx == pstep.end_idx:
            # The model chose to stop. If bits remain (pos < len) this turn
            # carries only a prefix of the payload and the rest continues in a
            # later real turn; if the payload ran out on this very choice the
            # turn is complete. The end token itself carries no surface bytes.
            ended_early = pos < len(full_bits)
            break

        chosen = pstep.reps[sym_idx]
        cur.append(chosen)
        msg_step += 1
        model.eval([chosen])
        lg = model.get_logits()

    complete = not ended_early
    consumed = min(pos, len(full_bits)) - start

    tail_step = 0
    if complete:
        # Payload done this turn: complete the sentence (the tail carries no
        # payload bits, only secure pseudo-random bits, and the receiver never
        # needs to mirror it). A cleanly-ended payload gets no tail at all.
        tail_bit_reader = _make_tail_bit_generator(cfg.crypto_key, cur)
        tail_min = getattr(cfg, "tail_min", 0)
        tail_max = getattr(cfg, "tail_max", 30)
        sentence_enders = getattr(cfg, "sentence_enders", ['.', '!', '?', '\n'])

        while tail_step < tail_max:
            if tail_step >= tail_min:
                recent = model.detokenize(cur[max(cover_start, len(cur) - 6):]).rstrip()
                last = recent[-1] if recent else ""
                if last in sentence_enders:
                    break
                if recent and not any(c.isalnum() for c in recent):
                    break

            pstep = sp.step(lg, cur, allow_end=_end_allowed(msg_step))
            cdf   = dc.build_cdf(pstep.masses)
            sym_idx, nb = dc.embed(cdf, cur, tail_bit_reader)
            tail_step += 1
            if sym_idx == pstep.end_idx:
                break                                   # the model ended its turn
            chosen = pstep.reps[sym_idx]
            cur.append(chosen)
            msg_step += 1
            model.eval([chosen])
            lg = model.get_logits()

    span = (cover_start, len(cur))
    cover = model.detokenize(cur[cover_start:len(cur)], skip_special=True)
    problem = _cover_problem(cover, cur[cover_start:len(cur)], sp, cfg)
    return cover, problem, step, tail_step, consumed, complete, cur, span


def _extract_turn(messages: list[dict], cover: str, model: LlamaCppModel,
                  cfg: StegoConfig) -> list[int]:
    """Bits recovered from one real turn's cover message, in order. `messages`
    is the real conversation BEFORE this cover (identical on both endpoints)."""
    prompt = model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    p_ids  = model.tokenize(prompt, add_bos=False, special=True)
    cur    = p_ids.copy()

    model.reset(); model.eval(p_ids)
    lg = model.get_logits()
    sp = _make_syncpool(model, cfg)
    dc = Discop(key=cfg.crypto_key, precision=32)

    cover_bytes = cover.encode("utf-8")
    extracted: list[int] = []
    pos = 0; msg_step = 0

    while True:
        pstep = sp.step(lg, cur, allow_end=_end_allowed(msg_step))
        cdf   = dc.build_cdf(pstep.masses)

        if pos >= len(cover_bytes):
            # Text over: the sender ended the turn here. If end-of-turn was a
            # candidate, recover the bits that choice carried -- the turn's
            # final payload bits, or (on the last turn) harmless tail bits the
            # AC body decoder ignores. If it was not a candidate, the turn's
            # bits were all in the cover tokens already read.
            if pstep.end_idx >= 0:
                bits, _ = dc.extract(cdf, cur, pstep.end_idx)
                extracted.extend(bits or [])
            break

        m = SyncPool.match(pstep, cover_bytes, pos)
        if m is None:
            raise ValueError("desync — cover does not match the model distribution")

        sym_idx, rep_id, nbytes = m
        bits, _ = dc.extract(cdf, cur, sym_idx)
        if bits is None:
            raise ValueError("tree desync — refusing to return garbage")

        extracted.extend(bits)
        pos += nbytes; cur.append(rep_id); msg_step += 1
        model.eval([rep_id]); lg = model.get_logits()

    return extracted


def _payload_bits(secret: str, codec: LLMTextCodec, cfg: StegoConfig):
    """CRC-prefixed, AC-coded payload bits for `secret` (no length header)."""
    raw_bits, _ = codec.encode_bits(secret)
    return crc_bits(secret.encode("utf-8")) + raw_bits, len(raw_bits)


def _full_bits(payload: list[int], ctx: bytes, counter: int, cfg: StegoConfig) -> list[int]:
    """Wire bits for one transmission: the masked attempt counter in the clear,
    then the keystream-encrypted payload. Keyed once by the first turn's context
    + counter, so the stream is continuous across however many turns carry it."""
    attempt_bits = getattr(cfg, "attempt_bits", 3)
    mask = _attempt_mask(cfg.crypto_key, ctx, attempt_bits)
    clear = [b ^ m for b, m
             in zip(unpack_bits(counter.to_bytes(4, "big"), 32)[-attempt_bits:], mask)]
    return clear + stream_cipher(payload, cfg.crypto_key, _derive_nonce(ctx, counter))


def _decode_payload(bits: list[int], ctx: bytes, codec: LLMTextCodec,
                    cfg: StegoConfig) -> Optional[str]:
    """Try to recover the secret from accumulated wire bits. Returns the secret
    if the CRC verifies, else None (payload not yet complete, or corrupt)."""
    attempt_bits = getattr(cfg, "attempt_bits", 3)
    if len(bits) < attempt_bits + CRC_BITS:
        return None
    mask     = _attempt_mask(cfg.crypto_key, ctx, attempt_bits)
    clear    = [b ^ m for b, m in zip(bits[:attempt_bits], mask)]
    counter  = int("".join(map(str, clear)), 2) if clear else 0
    dec_bits = stream_cipher(bits[attempt_bits:], cfg.crypto_key, _derive_nonce(ctx, counter))
    crc_recv = dec_bits[:CRC_BITS]
    secret   = codec.decode_bits(dec_bits[CRC_BITS:])
    return secret if check_crc(secret.encode("utf-8"), crc_recv) else None


class StegoSender:
    """Carries one secret across however many real assistant turns it takes.

    Call `next_cover(messages, model)` once per real turn, passing the real
    conversation so far (including the other party's latest message); it returns
    that turn's natural single-turn cover and raises StopIteration once the whole
    secret has been sent. The payload keystream is fixed on the first turn
    (keyed by that turn's context + a retried attempt counter); later turns embed
    the next slice of it, so they cannot re-roll the counter -- if a later turn's
    cover fails the sender-side round-trip check the transmission aborts."""

    def __init__(self, secret: str, codec: LLMTextCodec, cfg: StegoConfig):
        self.secret = secret
        self.cfg = cfg
        self._payload, self._body_bits = _payload_bits(secret, codec, cfg)
        self._full: Optional[list[int]] = None   # fixed on the first turn
        self._pos = 0
        self._turn = 0

    @property
    def done(self) -> bool:
        return self._full is not None and self._pos >= len(self._full)

    def next_cover(self, messages: list[dict], model: LlamaCppModel,
                   stats: Optional[dict] = None) -> str:
        if self.done:
            raise StopIteration("secret fully sent")

        if self._full is None:
            # First turn: pick the attempt counter here, retrying a fresh
            # keystream (and so a fresh cover) until one passes the round-trip
            # check -- nothing is committed to the wire yet.
            ctx = _context_digest(messages)
            self._ctx = ctx
            attempt_bits = getattr(self.cfg, "attempt_bits", 3)
            attempts = max(1, min(getattr(self.cfg, "gen_attempts", 3), 2 ** attempt_bits))
            print(f"  Secret: '{self.secret}'")
            print(f"  payload: {self._body_bits} body + {CRC_BITS} CRC + {attempt_bits} attempt bits")
            problem = None
            for attempt in range(1, attempts + 1):
                full = _full_bits(self._payload, ctx, attempt - 1, self.cfg)
                cover, problem, step, tail, consumed, complete, ids, span = \
                    _embed_turn(messages, full, 0, model, self.cfg)
                print(f"\n  Turn {self._turn + 1}: {step} tok "
                      f"({self._body_bits / max(step, 1):.2f} b/t) + {tail} tail, "
                      f"{'complete' if complete else 'partial'}")
                if problem is None:
                    self._full = full
                    self._pos = consumed
                    self._turn += 1
                    if stats is not None:
                        stats.update(ids=ids, spans=[span], payload_tokens=step,
                                     tail_tokens=tail, body_bits=self._body_bits,
                                     total_bits=len(full), attempt=attempt,
                                     complete=complete)
                    return cover
                print(f"  [!] attempt {attempt}/{attempts} rejected: {problem}")
            raise RuntimeError(f"no transmissible first cover after {attempts} attempts: {problem}")

        # Later turn: the keystream is fixed, so the cover is deterministic given
        # this turn's real context. No counter to re-roll.
        cover, problem, step, tail, consumed, complete, ids, span = \
            _embed_turn(messages, self._full, self._pos, model, self.cfg)
        print(f"\n  Turn {self._turn + 1}: {step} tok, {'complete' if complete else 'partial'}")
        if problem is not None:
            raise RuntimeError(
                f"turn {self._turn + 1} cover failed the round-trip check and "
                f"cannot be re-rolled mid-stream: {problem}")
        self._pos += consumed
        self._turn += 1
        if stats is not None:
            stats.update(ids=ids, spans=[span], payload_tokens=step,
                         tail_tokens=tail, body_bits=self._body_bits,
                         total_bits=len(self._full), complete=complete)
        return cover


class StegoReceiver:
    """Rebuilds a secret from the covers of successive real turns.

    Call `absorb(messages, cover, model)` once per real turn, passing the same
    real conversation the sender saw before that cover; it accumulates the turn's
    bits and returns True once the CRC verifies, after which `secret` holds the
    result. The transmission's context/nonce is pinned on the first absorb, so
    both endpoints must begin at the same point in the conversation."""

    def __init__(self, codec: LLMTextCodec, cfg: StegoConfig):
        self.codec = codec
        self.cfg = cfg
        self.secret: Optional[str] = None
        self.done = False
        self._bits: list[int] = []
        self._ctx: Optional[bytes] = None

    def absorb(self, messages: list[dict], cover: str, model: LlamaCppModel) -> bool:
        if self.done:
            return True
        if self._ctx is None:
            self._ctx = _context_digest(messages)

        self._bits.extend(_extract_turn(messages, cover, model, self.cfg))
        print()
        print("  LLM decompression …")
        secret = _decode_payload(self._bits, self._ctx, self.codec, self.cfg)
        if secret is not None:
            self.secret = secret
            self.done = True
        return self.done


def generate_oneshot(messages: list[dict], secret: str, model: LlamaCppModel,
                     codec: LLMTextCodec, cfg: StegoConfig,
                     stats: Optional[dict] = None) -> str:
    """Mode A -- the whole secret in a single assistant turn.

    Raises PayloadTooLong if the model ends its turn before the whole payload
    is embedded (the secret is too long for one message at these settings; use
    StegoSender / StegoReceiver instead). If `stats` is given it is filled for
    benchmarking (ids, spans, payload/tail token counts, bit counts, attempt)."""
    tx = StegoSender(secret, codec, cfg)
    cover = tx.next_cover(messages, model, stats=stats)
    if not tx.done:
        raise PayloadTooLong(
            "secret did not fit in one message — the model ended its turn with "
            "bits still due; use StegoSender / StegoReceiver for a conversation")
    return cover


def extract_oneshot(messages: list[dict], cover: str, model: LlamaCppModel,
                    codec: LLMTextCodec, cfg: StegoConfig) -> str:
    """Mode A receiver: recover the secret from a single-turn cover. `messages`
    is the context BEFORE the cover (the same one the sender used)."""
    rx = StegoReceiver(codec, cfg)
    if not rx.absorb(messages, cover, model):
        raise ValueError("CRC mismatch — corrupted, desynced, or an incomplete "
                         "multi-turn secret passed to the one-shot extractor")
    return rx.secret
