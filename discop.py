"""
Discop: recursive, distribution-preserving steganographic sampler.

Drop-in for the cover-stage sampler (same seam as SparSamp): takes a per-step
distribution -- here SyncPool's pool masses -- and returns which pool to emit
(encode) or which bits a pool encodes (decode). Unambiguous and ~entropy-rate.

Mechanism
---------
Build a Huffman tree over the pool masses (leaves = pools). Each internal node
is a bi-variate distribution over its two children. Walk root -> leaf; at every
internal node embed one bit via a "distribution copy" (rotation) of that node's
binary distribution, driven by a shared per-node uniform u.

Per-node embedding (smaller child s mass a, larger child g, m = a + b, a <= b):
  u in [0, m), shared (key + context + node id). For message bit b:
    copy 0:  s if u in [0, a)   else g
    copy 1:  s if u in [a, 2a)  else g          (2a <= m, so valid)
  A bit is embedded iff u < 2a; if u in [2a, m) the node is forced to g and
  carries nothing. The decoder knows u, so it knows whether a bit was carried,
  and -- given which child the emitted leaf's path took -- recovers it:
    u in [0, a):  taken == s -> 0,  taken == g -> 1
    u in [a, 2a): taken == g -> 0,  taken == s -> 1

Why the distribution is preserved (any message distribution):
    P(s) = P(u in [0,a) and b=0) + P(u in [a,2a) and b=1)
         = (a/m)*P(b=0) + (a/m)*P(b=1) = a/m, for ANY P(b).
  By induction over the tree the leaf marginal equals the model distribution,
  so the cover is statistically identical to honest sampling.

Why it is unambiguous: the two copies always send the two bit-values to
*different* children whenever u < 2a, and when u >= 2a no bit is in play and the
decoder sees that from u. No reliance on canonical re-tokenisation (SyncPool
handles that); no residual sampling ambiguity (unlike a single uniform grid).

Rate: ~0.9-0.95 of the per-step entropy (Huffman keeps node splits balanced).
The per-step cost is O(V log V) for the tree -- negligible next to an LLM
forward pass.
"""

from __future__ import annotations
import heapq
import hashlib

import numpy as np


class _Node:
    __slots__ = ("mass", "minleaf", "leaf", "c0", "c1")

    def __init__(self, mass, minleaf, leaf=None, c0=None, c1=None):
        self.mass = mass
        self.minleaf = minleaf     # min pool index in this subtree (unique per node)
        self.leaf = leaf           # pool index if a leaf, else None
        self.c0 = c0               # deterministically-ordered children
        self.c1 = c1


class Discop:
    def __init__(self, key: str, precision: int = 32):
        self.key = key.encode("utf-8") if isinstance(key, str) else key
        self.precision = precision
        self.T = 1 << precision

    # --- deterministic integer CDF over [0, T) (each width >= 1) -----------
    def build_cdf(self, masses) -> np.ndarray:
        p = np.maximum(np.asarray(masses, np.float64), 0.0)
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p)
            s = p.sum()
        p = p / s
        n = len(p)
        avail = self.T - n
        scaled = p * avail
        floors = np.floor(scaled).astype(np.int64)
        rem = int(avail - floors.sum())
        if rem > 0:
            order = np.argsort(-(scaled - floors), kind="stable")[:rem]
            floors[order] += 1
        widths = floors + 1
        cdf = np.zeros(n + 1, np.int64)
        np.cumsum(widths, out=cdf[1:])
        assert cdf[-1] == self.T
        return cdf

    # --- Huffman tree (identical on both sides via deterministic ties) ----
    def _build_tree(self, cdf: np.ndarray) -> _Node:
        widths = np.diff(cdf).astype(np.int64)
        n = len(widths)
        heap = []
        cnt = 0
        for i in range(n):
            heapq.heappush(heap, (int(widths[i]), i, cnt, _Node(int(widths[i]), i, leaf=i)))
            cnt += 1
        while len(heap) > 1:
            w1, l1, _, n1 = heapq.heappop(heap)
            w2, l2, _, n2 = heapq.heappop(heap)
            # order children by min-leaf so both sides agree which is c0 / c1
            if l1 <= l2:
                c0, c1, ml = n1, n2, l1
            else:
                c0, c1, ml = n2, n1, l2
            heapq.heappush(heap, (w1 + w2, ml, cnt, _Node(w1 + w2, ml, c0=c0, c1=c1)))
            cnt += 1
        return heap[0][3]

    # --- shared per-node uniform in [0, m) --------------------------------
    def _u(self, ctx_ids: list[int], node_id: int, m: int) -> int:
        seed = (self.key + b"|disc|"
                + np.asarray(ctx_ids, np.int64).tobytes()
                + b"|" + int(node_id).to_bytes(8, "big"))
        h = hashlib.sha256(seed).digest()
        return int.from_bytes(h[:16], "big") % m

    # --- ENCODE: message bits -> pool index -------------------------------
    def embed(self, cdf: np.ndarray, ctx_ids: list[int], read_bit):
        node = self._build_tree(cdf)
        nbits = 0
        while node.leaf is None:
            c0, c1 = node.c0, node.c1
            a0, a1 = c0.mass, c1.mass
            s, g, a = (c0, c1, a0) if a0 <= a1 else (c1, c0, a1)
            m = a0 + a1
            u = self._u(ctx_ids, node.minleaf, m)
            if u >= 2 * a:
                node = g                                  # forced, no bit
            else:
                b = read_bit() & 1
                nbits += 1
                if u < a:
                    node = s if b == 0 else g
                else:                                     # a <= u < 2a
                    node = g if b == 0 else s
        return node.leaf, nbits

    # --- DECODE: pool index -> message bits -------------------------------
    def extract(self, cdf: np.ndarray, ctx_ids: list[int], sym_idx: int):
        root = self._build_tree(cdf)
        bits: list[int] = []

        path = self._path_to(root, sym_idx)               # root->leaf path
        if path is None:
            return None, 0                                # desync guard

        for node, taken in path:
            c0, c1 = node.c0, node.c1
            a0, a1 = c0.mass, c1.mass
            s, g, a = (c0, c1, a0) if a0 <= a1 else (c1, c0, a1)
            m = a0 + a1
            u = self._u(ctx_ids, node.minleaf, m)
            if u >= 2 * a:
                continue                                  # no bit at this node
            if u < a:
                bits.append(0 if taken is s else 1)
            else:
                bits.append(0 if taken is g else 1)
        return bits, len(bits)

    @staticmethod
    def _path_to(node: _Node, target: int):
        """List of (internal_node, child_taken) from root to the target leaf."""
        if node.leaf is not None:
            return [] if node.leaf == target else None
        for child in (node.c0, node.c1):
            sub = Discop._path_to(child, target)
            if sub is not None:
                return [(node, child)] + sub
        return None
