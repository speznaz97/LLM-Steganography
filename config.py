from dataclasses import dataclass

@dataclass
class StegoConfig:
    stego_temp: float = 1.4231
    top_k: int = 104
    prob_threshold: float = 0.0051
    rep_penalty: float = 1.1243
    retoken_window: int = 10
    
    tail_max: int = 30
    tail_min: int = 1
    tail_temp: float = 0.6
    sentence_enders: frozenset = frozenset('.!?»)…')
    banned_chars: tuple = (
        '\n', '\r', '\t', '*', '#', '_', '[', ']',
        '<', '>', '|', '  ', '(', ')', '---'
    )
    header_bits: int = 16
    header_xor: int = 0x5555
    max_gen_tokens: int = 300