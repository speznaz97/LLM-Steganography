import numpy as np

class ArithmeticCoder:
    P     = 48
    FULL  = (1 << 48) - 1
    HALF  = 1 << 47
    QTR   = 1 << 46
    TOTAL = 1 << 26

    @staticmethod
    def build_cdf(probs: np.ndarray, total: int) -> np.ndarray:
        n = len(probs)
        p = np.maximum(np.asarray(probs, np.float64), 0.0)
        s = p.sum()
        if s <= 0:
            p = np.ones(n, np.float64); s = float(n)
        p /= s
        avail  = total - n
        scaled = p * avail
        floors = np.floor(scaled).astype(np.int64)
        gap    = int(avail - floors.sum())
        if gap > 0:
            idx = np.argpartition(-(scaled - floors), gap)[:gap]
            floors[idx] += 1
        elif gap < 0:
            removable = np.where(floors > 0)[0]
            order = removable[np.argsort(-floors[removable])]
            floors[order[:(-gap)]] -= 1
        cdf = np.empty(n + 1, np.int64)
        cdf[0] = 0
        np.cumsum(floors + 1, out=cdf[1:])
        assert cdf[-1] == total
        return cdf

    def narrow(self, lo, hi, cdf, sym, total):
        rng = hi - lo + 1
        new_hi = lo + (rng * int(cdf[sym + 1])) // total - 1
        new_lo = lo + (rng * int(cdf[sym]))      // total
        return new_lo, new_hi

    def find_symbol(self, lo, hi, val, cdf, total):
        rng = hi - lo + 1
        tgt = ((val - lo + 1) * total - 1) // rng
        return max(0, min(int(np.searchsorted(cdf, tgt, 'right')) - 1, len(cdf) - 2))

    def renorm_enc(self, lo, hi, pending):
        bits =[]
        while True:
            if hi < self.HALF:
                bits.append(0); bits.extend([1] * pending); pending = 0
            elif lo >= self.HALF:
                bits.append(1); bits.extend([0] * pending); pending = 0
                lo -= self.HALF; hi -= self.HALF
            elif lo >= self.QTR and hi < 3 * self.QTR:
                pending += 1; lo -= self.QTR; hi -= self.QTR
            else:
                break
            lo <<= 1; hi = (hi << 1) | 1
        return lo, hi, pending, bits

    def renorm_dec(self, lo, hi, val, read_bit):
        while True:
            if hi < self.HALF:
                pass
            elif lo >= self.HALF:
                val -= self.HALF; lo -= self.HALF; hi -= self.HALF
            elif lo >= self.QTR and hi < 3 * self.QTR:
                val -= self.QTR; lo -= self.QTR; hi -= self.QTR
            else:
                break
            lo <<= 1; hi = (hi << 1) | 1; val = (val << 1) | read_bit()
        return lo, hi, val

    def encode(self, symbols, cdfs, total):
        lo, hi, pending = 0, self.FULL, 0
        all_bits =[]
        for i, sym in enumerate(symbols):
            lo, hi = self.narrow(lo, hi, cdfs[i], sym, total)
            lo, hi, pending, new = self.renorm_enc(lo, hi, pending)
            all_bits.extend(new)
        pending += 1
        if lo < self.QTR:
            all_bits.append(0); all_bits.extend([1] * pending)
        else:
            all_bits.append(1); all_bits.extend([0] * pending)
        return all_bits

    def decode(self, bits, num_bits, cdf_fn, total, eos_id, max_len=2048):
        lo, hi, pos = 0, self.FULL, 0
        def rb():
            nonlocal pos
            b = bits[pos] if pos < num_bits else 0; pos += 1; return b
        val = 0
        for _ in range(self.P):
            val = (val << 1) | rb()
        result =[]
        for _ in range(max_len):
            c   = cdf_fn(result)
            sym = self.find_symbol(lo, hi, val, c, total)
            if sym == eos_id:
                break
            result.append(sym)
            lo, hi = self.narrow(lo, hi, c, sym, total)
            lo, hi, val = self.renorm_dec(lo, hi, val, rb)
        return result