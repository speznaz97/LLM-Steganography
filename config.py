from dataclasses import dataclass

@dataclass
class StegoConfig:
    # ---- cover stage: SyncPool (disambiguation) + Discop (sampler) ----
    stego_temp: float = 1.0          # cover sampling temperature; the main
                                     # capacity <-> naturalness dial
    syncpool_topk: int = 256         # candidate set size cap before pooling
    syncpool_floor: float = 1e-5     # drop candidates below this probability
    syncpool_minp: float = 0.0       # drop candidates below min_p * p_max (T=1);
                                     # 0 = off (top-k / floor only)

    # rep_penalty is DEPRECATED. A repetition penalty distorts the token
    # distribution and so breaks Discop's distribution-preservation guarantee
    # (the whole reason for switching away from the arithmetic coder). It is
    # kept at 1.0 -- a no-op -- only so any stale reference stays harmless.
    rep_penalty: float = 1.0

    crypto_key: str = "shared_secret_password_123"   # TODO: Diffie-Hellman exchange

    # ---- per-message nonce: derived from the conversation (both sides already
    #      hold it, and it differs per message) + a small attempt counter,
    #      embedded masked as the first cover bits; keys the payload keystream ----
    attempt_bits: int = 3            # counter width; caps retries at 2**attempt_bits
    gen_attempts: int = 3            # retries (new counter -> new nonce/cover) if
                                     # the cover fails the sender-side round-trip check

    # ---- end-of-turn: one turn produces exactly one message, always ending
    #      where the model chooses. A secret too long for a single message is
    #      carried across real turns by StegoSender / StegoReceiver instead. ----

    # ---- tail: completes the sentence after the payload (carries no bits);
    #      ends early if the model picks end-of-turn ----
    tail_max: int = 30
    tail_min: int = 0                # tail tokens forced before the stop check;
                                     # 0 = no tail if the payload ends a sentence
    sentence_enders: frozenset = frozenset('.!?»)…')

    # ---- surface bytes kept out of the candidate pool (applied both sides) ----
    banned_chars: tuple = (
        '\n', '\r', '\t', '*', '#', '_', '[', ']',
        '<', '>', '|', '  ', '(', ')', '---'
    )

    max_gen_tokens: int = 512