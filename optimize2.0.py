import struct
import math

import optuna
from optuna.samplers import TPESampler
import numpy as np

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from utils import np_softmax
from stego import generate_stego, extract_stego

# Adjust to your local files.
MODEL_PATH       = "Qwen3.5-4B-Q6_K.gguf"
CODEC_MODEL_PATH = "Qwen3.5-0.8B-Q8_0.gguf"

# True  -> two objectives: maximize bits/token, minimize perplexity (Pareto front)
# False -> one objective:  maximize bits/token (== shortest normalized cover)
MULTI_OBJECTIVE = True

N_TRIALS = 50


def compute_perplexity(model: LlamaCppModel, prompt_ids: list[int],
                       cover_ids: list[int]) -> float:
    """Perplexity of the cover under the base LLM at T=1. Lower = more natural.

    Note: because Discop preserves the distribution, the cover is a faithful
    sample of the model at `stego_temp`, so higher stego_temp -> lower-prob
    tokens chosen -> higher perplexity here. This metric is therefore a proxy
    for the *naturalness cost of temperature*, not a detectability signal
    (the scheme is distribution-preserving regardless of this number).
    """
    model.reset()
    model.eval(prompt_ids)
    nll = 0.0
    for tok in cover_ids:
        probs = np_softmax(model.get_logits(), temperature=1.0)
        nll -= math.log(max(float(probs[tok]), 1e-10))
        model.eval([tok])
    return math.exp(nll / max(1, len(cover_ids)))


def objective(trial, model, codec):
    cfg = StegoConfig(
        stego_temp     = trial.suggest_float("stego_temp", 0.8, 1.6, step=0.02),
        syncpool_topk  = trial.suggest_int("syncpool_topk", 32, 512, log=True),
        syncpool_floor = 1e-5,#trial.suggest_float("syncpool_floor", 1e-6, 1e-3, log=True),
        crypto_key     = "shared_secret_password_123",
        max_gen_tokens = 512,
    )

    total_bpt = 0.0
    total_ppl = 0.0
    n = len(EVAL_SUITE)

    for case in EVAL_SUITE:
        try:
            cover     = generate_stego(case["messages"], case["secret"], model, codec, cfg)
            recovered = extract_stego(case["messages"], cover, model, codec, cfg)
            if recovered.strip() != case["secret"].strip():
                # a config that cannot round-trip is invalid, not just bad
                #raise optuna.TrialPruned(f"extraction failed: {case['name']}")
                if recovered.strip() != case["secret"].strip():
                    print(f"\n  MISMATCH on {case['name']}:")
                    print(f"    sent:      {case['secret']!r}")
                    print(f"    recovered: {recovered!r}")
                    raise optuna.TrialPruned(f"extraction failed: {case['name']}")

            # payload bits = compressed-secret length prefix (cached -> ~free)
            wire = codec.encode(case["secret"])[0]
            payload_bits = struct.unpack(">I", wire[:4])[0]

            # cover length in tokens, INCLUDING the tail (it's honest overhead)
            cover_ids    = model.tokenize(cover, add_bos=False, special=False)
            cover_tokens = max(1, len(cover_ids))
            total_bpt += payload_bits / cover_tokens

            prompt     = model.apply_chat_template(case["messages"], add_generation_prompt=True)
            prompt_ids = model.tokenize(prompt, add_bos=False, special=True)
            total_ppl += compute_perplexity(model, prompt_ids, cover_ids)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"error on {case['name']}: {e}")

    avg_bpt = total_bpt / n
    avg_ppl = total_ppl / n
    print(f"\n[Trial {trial.number}] bits/tok={avg_bpt:.3f}  ppl={avg_ppl:.2f}  "
          f"(T={cfg.stego_temp:.2f}  topk={cfg.syncpool_topk}  "
          f"floor={cfg.syncpool_floor:.1e})")

    return (avg_bpt, avg_ppl) if MULTI_OBJECTIVE else avg_bpt


if __name__ == "__main__":
    DB_NAME = "stego_study.db"
    storage_url = f"sqlite:///{DB_NAME}"

    print(f"Loading {MODEL_PATH} for Optuna optimization...")
    model = LlamaCppModel(MODEL_PATH, n_ctx=8192, n_gpu_layers=0)
    codec_model = LlamaCppModel(CODEC_MODEL_PATH, n_ctx=2048, n_gpu_layers=0)
    codec = LLMTextCodec(codec_model, temperature=1.12)

    # --- compression cache: the codec params are fixed across trials, so the
    #     secret -> bits pass is identical every time. Compute it once. ---
    original_encode = codec.encode
    codec_cache: dict[str, tuple] = {}

    def cached_encode(text: str):
        if text not in codec_cache:
            codec_cache[text] = original_encode(text)
        return codec_cache[text]

    codec.encode = cached_encode

    EVAL_SUITE = [
        {
            "name": "Casual Short (WhatsApp)",
            "messages": [
                {"role": "system", "content": "You are a close friend chatting on WhatsApp. Use casual English, stay brief, and be friendly. Do not use emojis."},
                {"role": "user", "content": "Hey! Are we still on for coffee later?"},
            ],
            "secret": "Yes, 5pm works.",
        },
        {
            "name": "Planning Medium (Telegram)",
            "messages": [
                {"role": "system", "content": "You are a helpful friend on Telegram. Answer in one natural paragraph. No formal greetings. Do not use emojis."},
                {"role": "user", "content": "I'm looking for a good place to grab dinner tonight. Any suggestions?"},
            ],
            "secret": "The package is hidden behind the loose brick in the garden wall.",
        },
        {
            "name": "Work Informal (Slack)",
            "messages": [
                {"role": "system", "content": "You are a coworker on Slack. Be professional but concise. Use one or two sentences maximum. Do not use emojis."},
                {"role": "user", "content": "Did you get a chance to look at that spreadsheet I sent over this morning?"},
            ],
            "secret": "Use the secondary encryption key: 8842-Alpha-Niner-X.",
        },
        {
            "name": "Long Narrative (Catching Up)",
            "messages": [
                {"role": "system", "content": "You are catching up with an old friend via messenger. Write a warm, natural paragraph about a weekend trip. Do not use emojis."},
                {"role": "user", "content": "It's been forever! How was your trip to the mountains?"},
            ],
            "secret": "The meeting is compromised. Do not go to the safehouse. Proceed directly to the airport and wait for the courier near Gate B7.",
        },
    ]

    directions = ["maximize", "minimize"] if MULTI_OBJECTIVE else None
    direction  = None if MULTI_OBJECTIVE else "maximize"

    study = optuna.create_study(
        study_name="stego_pareto" if MULTI_OBJECTIVE else "stego_bpt",
        storage=storage_url,
        directions=directions,
        direction=direction,
        load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True),
    )

    print(f"\n[Storage] {DB_NAME}   [Dashboard] optuna-dashboard {storage_url}")
    print("Starting optimization...\n")
    study.optimize(lambda t: objective(t, model, codec), n_trials=N_TRIALS, n_jobs=1)

    print("\n" + "=" * 60)
    print("OPTIMIZATION FINISHED")
    print("=" * 60)

    if MULTI_OBJECTIVE:
        print("\nPareto-optimal configs (bits/token higher is better, ppl lower is better):")
        front = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
        for t in front:
            bpt, ppl = t.values
            p = t.params
            print(f"  bits/tok={bpt:5.3f}  ppl={ppl:6.2f}  |  "
                  f"stego_temp={p['stego_temp']:.2f}  "
                  f"syncpool_topk={p['syncpool_topk']}  ")
                  #f"syncpool_floor={p['syncpool_floor']:.1e}")
        print("\nPick the knee of this front: the point past which more "
              "bits/token costs a sharp jump in perplexity.")
    else:
        best = study.best_trial
        print(f"\nBest (Trial {best.number}): bits/token = {best.value:.3f}")
        for k, v in best.params.items():
            print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")