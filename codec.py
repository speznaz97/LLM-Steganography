import struct
import time
import numpy as np

from llm import LlamaCppModel
from arithmetic import ArithmeticCoder
from utils import np_softmax, pack_bits, unpack_bits

class LLMTextCodec:
    def __init__(self, model: LlamaCppModel, temperature: float = 1.0):
        self.model       = model
        self.temperature = temperature
        self.ac          = ArithmeticCoder()
        self.total       = self.ac.TOTAL
        self.eos_id      = model.eos_id
        self.bos_id      = model.bos_id
        print(f"  [Codec] vocab={model.n_vocab}  bos={self.bos_id}  eos={self.eos_id}  T={temperature}")

    def _logits_to_cdf(self, logits: np.ndarray) -> np.ndarray:
        return self.ac.build_cdf(np_softmax(logits, self.temperature), self.total)

    def encode(self, text: str):
        tokens  = self.model.tokenize(text, add_bos=False, special=False)
        symbols = tokens + [self.eos_id]

        self.model.reset()
        self.model.eval([self.bos_id])
        t0   = time.time()
        cdfs = [self._logits_to_cdf(self.model.get_logits())]

        for tok in tokens:
            self.model.eval([tok]) 
            cdfs.append(self._logits_to_cdf(self.model.get_logits()))

        bits = self.ac.encode(symbols, cdfs, self.total)
        wire = struct.pack('>I', len(bits)) + pack_bits(bits)
        return wire, dict(tokens=len(tokens), bits=len(bits), wire=len(wire), time=time.time() - t0)

    def decode(self, wire: bytes) -> str:
        num_bits = struct.unpack('>I', wire[:4])[0]
        bits     = unpack_bits(wire[4:], num_bits)

        self.model.reset()
        self.model.eval([self.bos_id])
        first_cdf = self._logits_to_cdf(self.model.get_logits())

        def cdf_fn(ctx):
            if not ctx: return first_cdf
            self.model.eval([ctx[-1]])
            return self._logits_to_cdf(self.model.get_logits())

        ids = self.ac.decode(bits, num_bits, cdf_fn, self.total, eos_id=self.eos_id, max_len=2048)
        return self.model.detokenize(ids, skip_special=True)