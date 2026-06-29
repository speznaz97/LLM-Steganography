"""
SyncPool: distribution-preserving disambiguation for LLM arithmetic / sampling
steganography.

Problem it solves
-----------------
Sub-word (BPE) tokenisation is many-to-one over byte strings: the same surface
text can be produced by different token sequences. When the sender emits token
sequence S but the receiver re-tokenises the surface text into a different
sequence S', the per-step probability model diverges and the hidden payload
decodes to garbage. The original `_safe_probs` defended against this by
*filtering out* any candidate whose emission would change the canonical
tokenisation. That is lossy (it discards probability mass / entropy) and it is
NOT distribution-preserving (it changes the per-token distribution the model is
supposed to emit) -- which is exactly what a statistical detector keys on.

SyncPool instead *groups* mutually-ambiguous tokens into pools and embeds the
secret at the POOL level, leaving the original token distribution untouched.

Algorithm (run identically by sender and receiver every step)
-------------------------------------------------------------
1. Take the model's next-token distribution and a reproducible truncation
   (top-k + probability floor + banned/special removal).
2. Group survivors into pools = connected components under the byte-string
   "is-a-prefix-of" relation. Two tokens land in the same pool iff one's
   surface bytes are a prefix of the other's -- precisely the case the receiver
   cannot tell apart from the bytes alone.
3. Pool mass P(pool) = sum of member probabilities.
4. For each pool, pick a *representative* token via a key-seeded PRNG that
   samples within the pool according to the renormalised model probabilities.
   Both parties derive the same representative because they share the key and
   the (identical) context.

Sender:   embed bits by choosing a pool with the secure sampler over P(pool),
          then emit that pool's representative token.
Receiver: at the current byte offset, exactly one pool's representative is a
          prefix of the remaining cover bytes (representatives of distinct
          pools are provably prefix-free); that identifies the pool -> bits,
          and its byte length advances the offset.

Why it stays byte-aligned: because the representative is reproducible, the
receiver reconstructs the *same* token id the sender emitted, appends the *same*
id to its context, and advances by the *same* number of bytes -- so the two
contexts never diverge and nothing depends on canonical re-tokenisation.

Why the distribution is preserved: marginalising over the PRNG,
P(emit t) = P(pool(t)) * P(t | pool(t)) = P(t). The pool-level sampler (your
ArithmeticCoder) preserves the pool marginal, so the token marginal is the
untouched model distribution.

Assumptions / caveats
----------------------
* Assumes byte-level BPE (GPT-2 / Qwen family), where a token maps to a fixed
  byte string independent of context. SentencePiece models (Llama) render
  pieces in context -- adapt `_token_bytes` to detokenise in context if so.
* Truncation (top-k / floor) is a small, reproducible deviation from the true
  support, applied identically on both sides -- the same compromise every
  nucleus-style secure scheme makes. Lower `prob_floor` / raise `top_k` for
  stricter security at the cost of speed.
* The representative PRNG re-hashes the context each step (O(context)). For long
  generations swap in a rolling hash; correctness only needs it reproducible.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass

import numpy as np

from utils import np_softmax


@dataclass
class PoolStep:
    reps: list[int]           # representative token id per pool, in cdf order
    rep_bytes: list[bytes]    # representative surface bytes, same order
    masses: np.ndarray        # pool probability mass, same order (sums to ~1)
    members: list[list[int]]  # member token ids per pool (inspection / debug)


class SyncPool:
    def __init__(self, model, key: str, temperature: float = 1.0,
                 top_k: int = 256, prob_floor: float = 1e-5,
                 banned_bytes: tuple[bytes, ...] = ()):
        self.model = model
        self.key = key.encode("utf-8") if isinstance(key, str) else key
        self.temperature = temperature
        self.top_k = top_k
        self.prob_floor = prob_floor
        self.banned = tuple(
            b.encode("utf-8") if isinstance(b, str) else b for b in banned_bytes
        )
        self._byte_cache: dict[int, bytes] = {}

    # --- token -> fixed surface bytes (byte-level BPE) ----------------------
    def _token_bytes(self, tok: int) -> bytes:
        b = self._byte_cache.get(tok)
        if b is None:
            # underlying llama detokenize returns raw bytes (no lossy decode)
            b = bytes(self.model.llm.detokenize([tok]))
            self._byte_cache[tok] = b
        return b

    # --- reproducible within-pool representative ---------------------------
    def _representative(self, ids: list[int], probs: np.ndarray,
                        ctx_ids: list[int]) -> int:
        if len(ids) == 1:
            return ids[0]
        order = np.argsort(np.asarray(ids, np.int64))   # deterministic order
        ids_s = [ids[i] for i in order]
        p = np.asarray(probs, np.float64)[order]
        s = p.sum()
        p = p / s if s > 0 else np.full(len(ids_s), 1.0 / len(ids_s))
        # uniform in [0, 1) derived from key + context: identical on both sides
        seed = self.key + b"|rep|" + np.asarray(ctx_ids, np.int64).tobytes()
        h = hashlib.sha256(seed).digest()
        u = int.from_bytes(h[:8], "big") / 2.0**64
        c = np.cumsum(p)
        j = int(np.searchsorted(c, u, "right"))
        return ids_s[min(j, len(ids_s) - 1)]

    # --- prefix-closure grouping (connected components) --------------------
    @staticmethod
    def _group(strs: list[bytes]) -> list[list[int]]:
        n = len(strs)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # In lexicographic order, every string having strs[i] as a prefix is
        # contiguous and immediately after i, so a forward scan-until-mismatch
        # captures all direct prefix relations; union-find handles transitivity.
        order = sorted(range(n), key=lambda i: strs[i])
        for a in range(n):
            i = order[a]
            si = strs[i]
            for b in range(a + 1, n):
                j = order[b]
                if strs[j].startswith(si):
                    union(i, j)
                else:
                    break
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    # --- main per-step computation -----------------------------------------
    def step(self, logits: np.ndarray, ctx_ids: list[int]) -> PoolStep:
        probs = np_softmax(logits, self.temperature)

        # reproducible truncation: top-k by probability, then a floor
        k = min(self.top_k, probs.shape[0])
        cand = np.argpartition(probs, -k)[-k:]
        cand = cand[probs[cand] >= self.prob_floor]

        specials = self.model.special_ids
        ids: list[int] = []
        strs: list[bytes] = []
        ps: list[float] = []
        for iv in cand.tolist():
            if iv in specials:
                continue
            b = self._token_bytes(iv)
            if not b:
                continue
            if any(bad in b for bad in self.banned):
                continue
            ids.append(iv)
            strs.append(b)
            ps.append(float(probs[iv]))

        # fallback: never emit nothing (mirrors the original safeguard)
        if not ids:
            top = int(np.argmax(probs))
            return PoolStep([top], [self._token_bytes(top)],
                            np.array([1.0]), [[top]])

        comps = self._group(strs)

        reps: list[int] = []
        rep_bytes: list[bytes] = []
        masses: list[float] = []
        members: list[list[int]] = []
        for comp in comps:
            m_ids = [ids[i] for i in comp]
            m_ps = np.array([ps[i] for i in comp], np.float64)
            rep = self._representative(m_ids, m_ps, ctx_ids)
            reps.append(rep)
            rep_bytes.append(self._token_bytes(rep))
            masses.append(float(m_ps.sum()))
            members.append(m_ids)

        # deterministic, identical ordering on both sides
        idx = sorted(range(len(reps)), key=lambda i: (rep_bytes[i], reps[i]))
        reps = [reps[i] for i in idx]
        rep_bytes = [rep_bytes[i] for i in idx]
        masses_arr = np.array([masses[i] for i in idx], np.float64)
        members = [members[i] for i in idx]
        masses_arr /= masses_arr.sum()
        return PoolStep(reps, rep_bytes, masses_arr, members)

    # --- receiver: surface bytes -> pool index -----------------------------
    @staticmethod
    def match(step: PoolStep, cover: bytes, pos: int):
        """Return (pool_index, rep_id, nbytes), or None on desync.

        Representatives of distinct pools are prefix-free, so at most one is a
        prefix of cover[pos:]; the longest-match tie-break is purely defensive.
        """
        best = -1
        best_len = -1
        tail = cover[pos:]
        for i, rb in enumerate(step.rep_bytes):
            if rb and tail.startswith(rb) and len(rb) > best_len:
                best, best_len = i, len(rb)
        if best < 0:
            return None
        return best, step.reps[best], best_len
