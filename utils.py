import numpy as np
import hashlib

def np_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.float64(logits) / temperature
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()

def np_topk(arr: np.ndarray, k: int):
    k = min(k, len(arr))
    idx = np.argpartition(arr, -k)[-k:]
    order = np.argsort(-arr[idx])
    idx = idx[order]
    return arr[idx], idx

def pack_bits(bits: list[int]) -> bytes:
    pad = (-len(bits)) % 8
    return bytes(np.packbits(np.array(bits + [0] * pad, dtype=np.uint8)))

def unpack_bits(data: bytes, num_bits: int) -> list[int]:
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8)
    )[:num_bits].tolist()


def stream_cipher(bits: list[int], key: str, nonce: bytes) -> list[int]:
    """Lightweight CTR stream cipher to guarantee uniformly random payload bits.
    The nonce must be unique per message, or two payloads XOR to plaintext XOR."""
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    out_bits = []
    counter = 0
    while len(out_bits) < len(bits):
        # Generate a deterministic pseudo-random block (fixed-width fields, so
        # no (nonce, counter) pair can alias another)
        block = hashlib.sha256(key_bytes + b"|ctr|" + nonce
                               + counter.to_bytes(8, "big")).digest()
        for byte in block:
            for i in range(8):
                out_bits.append((byte >> (7 - i)) & 1)
        counter += 1
    
    # XOR the payload with the keystream
    return [b ^ k for b, k in zip(bits, out_bits[:len(bits)])]