import optuna
import math
import numpy as np
import time

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from utils import np_softmax
from stego import generate_stego, extract_stego

MODEL_PATH = "LFM2-8B-A1B-Q6_K.gguf"

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
    # Search Space
    cfg = StegoConfig(
        stego_temp=trial.suggest_float("stego_temp", 1.0, 1.5),
        top_k=trial.suggest_int("top_k", 40, 120),
        prob_threshold=trial.suggest_float("prob_threshold", 0.002, 0.02, log=True),
        rep_penalty=trial.suggest_float("rep_penalty", 1.05, 1.2),
        retoken_window=trial.suggest_int("retoken_window", 4, 10),
        max_gen_tokens=400 # Per-message limit
    )

    total_tokens = 0
    total_perplexity = 0
    count = len(EVAL_SUITE)

    for case in EVAL_SUITE:
        try:
            # 1. Generate
            cover = generate_stego(case["messages"], case["secret"], model, codec, cfg)
            
            # 2. Extract & Verify
            recovered = extract_stego(case["messages"], cover, model, codec, cfg)
            if recovered.strip() != case["secret"].strip():
                raise optuna.TrialPruned(f"Extraction failed on: {case['name']}")

            # 3. Measure Efficiency (Tokens)
            cover_ids = model.tokenize(cover, add_bos=False, special=False)
            total_tokens += len(cover_ids)

            # 4. Measure Readability (Perplexity)
            prompt = model.apply_chat_template(case["messages"], add_generation_prompt=True)
            prompt_ids = model.tokenize(prompt, add_bos=False, special=True)
            total_perplexity += compute_perplexity(model, prompt_ids, cover_ids)

        except (RuntimeError, Exception) as e:
            # If it takes too long or crashes, kill the trial
            raise optuna.TrialPruned(f"Error on {case['name']}: {e}")

    # Return the averages
    avg_tokens = total_tokens / count
    avg_ppl = total_perplexity / count
    
    print(f"\n[Trial {trial.number}] Avg Tokens: {avg_tokens:.1f}, Avg PPL: {avg_ppl:.2f}")
    return avg_tokens, avg_ppl

if __name__ == "__main__":
    # --- 1. Database Configuration ---
    # This file will store all trials so you can view them in the dashboard
    DB_NAME = "stego_study.db"
    storage_url = f"sqlite:///{DB_NAME}"

    print(f"Loading {MODEL_PATH} for Optuna optimization...")
    model = LlamaCppModel(
        MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=0, 
    )
    codec = LLMTextCodec(model, temperature=1.0)

    # Standardized test inputs
    EVAL_SUITE = [
        {
            "name": "Casual Short (WhatsApp)",
            "messages": [
                {"role": "system", "content": "You are a close friend chatting on WhatsApp. Use casual English, stay brief, and be friendly."},
                {"role": "user", "content": "Hey! Are we still on for coffee later?"}
            ],
            "secret": "Yes, 5pm works." # Short secret (~40 bits)
        },
        {
            "name": "Planning Medium (Telegram)",
            "messages": [
                {"role": "system", "content": "You are a helpful friend on Telegram. Answer in one natural paragraph. No formal greetings."},
                {"role": "user", "content": "I'm looking for a good place to grab dinner tonight. Any suggestions?"}
            ],
            "secret": "The package is hidden behind the loose brick in the garden wall." # Medium secret (~150 bits)
        },
        {
            "name": "Work Informal (Slack)",
            "messages": [
                {"role": "system", "content": "You are a coworker on Slack. Be professional but concise. Use one or two sentences maximum."},
                {"role": "user", "content": "Did you get a chance to look at that spreadsheet I sent over this morning?"}
            ],
            "secret": "Use the secondary encryption key: 8842-Alpha-Niner-X." # Specific data secret (~120 bits)
        },
        {
            "name": "Long Narrative (Catching Up)",
            "messages": [
                {"role": "system", "content": "You are catching up with an old friend via messenger. Write a warm, natural paragraph about a weekend trip."},
                {"role": "user", "content": "It's been forever! How was your trip to the mountains?"}
            ],
            "secret": "The meeting is compromised. Do not go to the safehouse. Proceed directly to the airport and wait for the courier near Gate B7." # Long secret (~300+ bits)
        }
    ]

    # --- 2. Create or Load Study with Storage ---
    study = optuna.create_study(
        study_name="robust_averaging",
        storage=storage_url,
        directions=["minimize", "minimize"],
        load_if_exists=True
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
    
    # Extract Pareto front (the best trade-offs between efficiency and readability)
    pareto_trials = study.best_trials
    print(f"\nFound {len(pareto_trials)} Pareto-optimal configurations:")
    
    for i, trial in enumerate(pareto_trials, 1):
        print(f"\n[Trade-off Option {i} - Trial {trial.number}]")
        print(f"  Efficiency:  {trial.values[0]} tokens")
        print(f"  Readability: {trial.values[1]:.2f} perplexity")
        print("  Config Parameters to apply to StegoConfig:")
        for key, value in trial.params.items():
            if isinstance(value, float):
                print(f"    {key} = {value:.4f}")
            else:
                print(f"    {key} = {value}")

    print("\n💡 TIP: Pick a config with the lowest tokens where perplexity is still below ~15.0 for human-like text.")