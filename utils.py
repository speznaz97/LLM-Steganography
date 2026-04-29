import numpy as np

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