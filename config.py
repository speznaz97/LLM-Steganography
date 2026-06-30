from dataclasses import dataclass

@dataclass
class StegoConfig:
    # ---- cover stage: SyncPool (disambiguation) + Discop (sampler) ----
    stego_temp: float = 1.0          # cover sampling temperature; the main
                                     # capacity <-> naturalness dial
    syncpool_topk: int = 256         # candidate set size before pooling
    syncpool_floor: float = 1e-5     # drop candidates below this probability

    # rep_penalty is DEPRECATED. A repetition penalty distorts the token
    # distribution and so breaks Discop's distribution-preservation guarantee
    # (the whole reason for switching away from the arithmetic coder). It is
    # kept at 1.0 -- a no-op -- only so any stale reference stays harmless.
    rep_penalty: float = 1.0

    crypto_key: str = "shared_secret_password_123"   # TODO: Diffie-Hellman exchange

    # ---- tail: completes the sentence after the payload (carries no bits) ----
    tail_max: int = 30
    tail_min: int = 1                # unused by the current tail loop; harmless
    sentence_enders: frozenset = frozenset('.!?»)…')

    # ---- surface bytes kept out of the candidate pool (applied both sides) ----
    banned_chars: tuple = (
        '\n', '\r', '\t', '*', '#', '_', '[', ']',
        '<', '>', '|', '  ', '(', ')', '---'
    )

    # ---- payload length header ----
    header_bits: int = 16
    header_xor: int = 0x5555

    max_gen_tokens: int = 512