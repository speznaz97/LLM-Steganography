import optuna
from optuna.samplers import TPESampler
import math
import numpy as np
import time

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from utils import np_softmax
from stego import generate_stego, extract_stego

MODEL_PATH = "LFM2.5-8B-A1B-UD-Q6_K.gguf"
CODEC_MODEL_PATH = "LFM2.5-230M-Q8_0.gguf"

def compute_perplexity(model: LlamaCppModel, prompt_ids: list[int], cover_ids: list[int]) -> float:
    """
    Computes the perplexity of the generated cover text under the base LLM.
    Lower perplexity -> More natural, human-like text.
    """
    model.reset()
    model.eval(prompt_ids)
    nll = 0.0

    for tok in cover_ids:
        logits = model.get_logits()
        # Use standard temperature 1.0 to judge how "natural" the token is
        probs = np_softmax(logits, temperature=1.0)
        p = probs[tok]
        
        # Accumulate Negative Log-Likelihood, avoiding log(0)
        nll -= math.log(max(p, 1e-10))
        
        # Advance context
        model.eval([tok])

    # Return e^(avg_nll)
    return math.exp(nll / max(1, len(cover_ids)))

def objective(trial, model, codec):
    # Search Space: Single-variable optimization of the rep_penalty
    cfg = StegoConfig(
        rep_penalty=trial.suggest_float("rep_penalty", 0.8, 1.3, step=0.01),
        retoken_window=6, # Hardcoded
        crypto_key="shared_secret_password_123",
        max_gen_tokens=400
    )

    total_perplexity = 0
    count = len(EVAL_SUITE)

    for case in EVAL_SUITE:
        try:
            # 1. Generate (Uses the cached/optimized codec)
            cover = generate_stego(case["messages"], case["secret"], model, codec, cfg)
            
            # 2. Extract & Verify
            recovered = extract_stego(case["messages"], cover, model, codec, cfg)
            if recovered.strip() != case["secret"].strip():
                raise optuna.TrialPruned(f"Extraction failed on: {case['name']}")

            # 3. Measure Readability (Perplexity)
            prompt = model.apply_chat_template(case["messages"], add_generation_prompt=True)
            prompt_ids = model.tokenize(prompt, add_bos=False, special=True)
            cover_ids = model.tokenize(cover, add_bos=False, special=False)
            
            ppl = compute_perplexity(model, prompt_ids, cover_ids)
            if ppl > 30.0: # Early stopping
                raise optuna.TrialPruned(f"Text too unnatural (PPL {ppl:.1f})")
                
            total_perplexity += ppl

        except (RuntimeError, Exception) as e:
            raise optuna.TrialPruned(f"Error on {case['name']}: {e}")

    # Return average perplexity
    avg_ppl = total_perplexity / count
    print(f"\n[Trial {trial.number}] Avg PPL: {avg_ppl:.2f}")
    return avg_ppl

if __name__ == "__main__":
    # --- 1. Database Configuration ---
    DB_NAME = "stego_study.db"
    storage_url = f"sqlite:///{DB_NAME}"

    print(f"Loading {MODEL_PATH} for Optuna optimization...")
    model = LlamaCppModel(
        MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=0, 
    )

    codec_model = LlamaCppModel(
        CODEC_MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=0, 
    )

    codec = LLMTextCodec(codec_model, temperature=1.12)

    # --- DYNAMIC COMPRESSION CACHE ---
    # Intercepts raw encode requests to bypass the compression pass after Trial 0
    original_encode = codec.encode
    codec_cache = {}

    def cached_encode(text: str):
        if text not in codec_cache:
            codec_cache[text] = original_encode(text)
        return codec_cache[text]

    codec.encode = cached_encode
    # ---------------------------------

    # Standardized test inputs
    EVAL_SUITE = [
        {
            "name": "Casual Short (WhatsApp)",
            "messages": [
                {"role": "system", "content": "You are a close friend chatting on WhatsApp. Use casual English, stay brief, and be friendly."},
                {"role": "user", "content": "Hey! Are we still on for coffee later?"}
            ],
            "secret": "Yes, 5pm works."
        },
        {
            "name": "Planning Medium (Telegram)",
            "messages": [
                {"role": "system", "content": "You are a helpful friend on Telegram. Answer in one natural paragraph. No formal greetings."},
                {"role": "user", "content": "I'm looking for a good place to grab dinner tonight. Any suggestions?"}
            ],
            "secret": "The package is hidden behind the loose brick in the garden wall."
        },
        {
            "name": "Work Informal (Slack)",
            "messages": [
                {"role": "system", "content": "You are a coworker on Slack. Be professional but concise. Use one or two sentences maximum."},
                {"role": "user", "content": "Did you get a chance to look at that spreadsheet I sent over this morning?"}
            ],
            "secret": "Use the secondary encryption key: 8842-Alpha-Niner-X."
        },
        {
            "name": "Long Narrative (Catching Up)",
            "messages": [
                {"role": "system", "content": "You are catching up with an old friend via messenger. Write a warm, natural paragraph about a weekend trip."},
                {"role": "user", "content": "It's been forever! How was your trip to the mountains?"}
            ],
            "secret": "The meeting is compromised. Do not go to the safehouse. Proceed directly to the airport and wait for the courier near Gate B7."
        }
    ]

    # --- 2. Create or Load Study with Storage ---
    study = optuna.create_study(
        study_name="optimize_embedder",
        storage=storage_url,
        direction="minimize", 
        load_if_exists=True,
        sampler=TPESampler(seed=42)
    )

    print(f"\n[Storage] Optimization data saved to: {DB_NAME}")
    print(f"[Dashboard] Run this in a new terminal to view progress:")
    print(f"    optuna-dashboard {storage_url}")
    print("\nStarting Optimization...")

    study.optimize(
        lambda t: objective(t, model, codec), 
        n_trials=50, 
        n_jobs=1 
    )

    print("\n" + "="*50)
    print("OPTIMIZATION FINISHED")
    print("="*50)
    
    # --- FIXED PRINT LOGIC FOR SINGLE-OBJECTIVE STUDY ---
    best_trial = study.best_trial
    print(f"\nBest configuration found (Trial {best_trial.number}):")
    print(f"  Perplexity: {best_trial.value:.2f}")
    print("  Config Parameters to apply to StegoConfig:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"    {key} = {value:.4f}")
        else:
            print(f"    {key} = {value}")

    print("\n💡 TIP: Pick the optimized 'rep_penalty' value and apply it to StegoConfig in config.py!")