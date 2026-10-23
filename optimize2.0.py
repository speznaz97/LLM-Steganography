import argparse
import math
import time
import unicodedata

import optuna
from optuna.samplers import TPESampler

from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from utils import np_softmax
from stego import generate_stego, extract_stego

# Adjust to your local files.
MODEL_PATH       = "Qwen3.5-4B-Q6_K.gguf"
CODEC_MODEL_PATH = "Qwen3.5-0.8B-Q8_0.gguf"
CODEC_TEMP       = 1.12

# True  -> two objectives: maximize bits/token, minimize perplexity (Pareto front)
# False -> one objective:  maximize bits/token (== shortest normalized cover)
MULTI_OBJECTIVE = True

N_TRIALS  = 30
# Generation is deterministic per (conversation, key), so each case is run
# under REPEATS different keys -> different keystreams -> different covers.
# Averages over those instead of scoring one lucky/unlucky sample per case.
REPEATS   = 2
TOPK_CAP  = 256          # top-k is only a safety cap now; min-p does the cutting
# Feasibility constraint: max fraction of runs whose cover used every allowed
# message (cfg.max_messages). In the last message end-of-turn is disallowed, so
# a capped cover can't stop where the model wants to -- the run-on text this
# whole scheme tries to avoid. Infeasible trials are kept off the Pareto front.
MAX_CAP_RATE = 0.10

STUDY_NAME = "stego_minp_v1"
DB_NAME    = "stego_study.db"

# Fixed configs scored with --baseline, for comparison with the tuned front
BASELINES = [
    {"stego_temp": 1.22, "syncpool_topk": 47, "syncpool_minp": 0.0},
    {"stego_temp": 1.38, "syncpool_topk": 35, "syncpool_minp": 0.0},
]

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


def compute_perplexity(model: LlamaCppModel, ids: list[int],
                       spans: list[tuple[int, int]]) -> tuple[float, int]:
    """Perplexity of the tokens actually sent, under the base LLM at T=1.

    `ids` is the exact sequence generate_stego produced (prompt, messages and
    the seams between them) and `spans` the slice of each message. SyncPool's
    tokens are not the canonical split of the cover text, so re-tokenizing the
    text would score a different sequence. Lower = more natural.

    Note: because Discop preserves the distribution, the cover is a faithful
    sample of the model at `stego_temp` (after truncation), so this measures
    the naturalness cost of the sampling settings, not detectability.
    """
    scored = set()
    for a, b in spans:
        scored.update(range(a, b))
    start = spans[0][0]

    model.reset()
    model.eval(ids[:start])
    nll, n = 0.0, 0
    for i in range(start, len(ids)):
        if i in scored:
            probs = np_softmax(model.get_logits(), temperature=1.0)
            nll -= math.log(max(float(probs[ids[i]]), 1e-10))
            n += 1
        model.eval([ids[i]])
    return math.exp(nll / max(1, n)), n


def emoji_rate(text: str) -> float:
    """Fraction of characters that are symbols/emoji (Unicode 'So')."""
    if not text:
        return 0.0
    return sum(unicodedata.category(c) == "So" for c in text) / len(text)


def evaluate(params: dict, model, codec, label: str = "") -> dict:
    """Run the whole suite REPEATS times under one config. Raises on any
    round-trip failure: a config that cannot round-trip is invalid, not just bad."""
    rows = []
    for r in range(REPEATS):
        cfg = StegoConfig(
            stego_temp     = params["stego_temp"],
            syncpool_topk  = params.get("syncpool_topk", TOPK_CAP),
            syncpool_minp  = params.get("syncpool_minp", 0.0),
            crypto_key     = f"bench-key-{r}",
            max_gen_tokens = 512,
        )
        for case in EVAL_SUITE:
            t0 = time.time()
            stats = {}
            covers    = generate_stego(case["messages"], case["secret"], model, codec, cfg, stats=stats)
            recovered = extract_stego(case["messages"], covers, model, codec, cfg)
            if recovered.strip() != case["secret"].strip():
                print(f"\n  MISMATCH on {case['name']} (repeat {r}):")
                print(f"    sent:      {case['secret']!r}")
                print(f"    recovered: {recovered!r}")
                raise RuntimeError(f"extraction failed: {case['name']}")

            ppl, n_tok = compute_perplexity(model, stats["ids"], stats["spans"])
            text = " ".join(covers)
            rows.append({
                "case": case["name"], "repeat": r,
                # bits of the secret per cover token sent, tail included (it's honest overhead)
                "bpt": stats["body_bits"] / max(1, n_tok),
                "ppl": ppl, "tokens": n_tok,
                "tail": stats["tail_tokens"], "msgs": len(covers),
                "emoji": emoji_rate(text), "retries": stats["attempt"] - 1,
                "capped": float(stats["capped"]),
                "secs": time.time() - t0, "covers": covers,
            })

    n = len(rows)
    summary = {k: sum(row[k] for row in rows) / n
               for k in ("bpt", "ppl", "tokens", "tail", "msgs", "capped", "emoji", "retries", "secs")}
    summary["rows"] = rows

    print(f"\n[{label}] bits/tok={summary['bpt']:.3f}  ppl={summary['ppl']:.2f}  "
          f"tok={summary['tokens']:.0f}  msgs={summary['msgs']:.2f}  "
          f"capped={summary['capped'] * 100:.0f}%  emoji={summary['emoji'] * 100:.1f}%  "
          f"(T={params['stego_temp']:.2f}  topk={params.get('syncpool_topk', TOPK_CAP)}  "
          f"minp={params.get('syncpool_minp', 0.0):.3f})")
    return summary


def objective(trial, model, codec):
    params = {
        "stego_temp":    trial.suggest_float("stego_temp", 0.9, 2.2, step=0.02),
        "syncpool_minp": trial.suggest_float("syncpool_minp", 0.005, 0.3, log=True),
        "syncpool_topk": TOPK_CAP,
    }
    try:
        s = evaluate(params, model, codec, label=f"Trial {trial.number}")
    except Exception as e:
        raise optuna.TrialPruned(str(e))

    for k in ("tokens", "tail", "msgs", "capped", "emoji", "retries", "secs"):
        trial.set_user_attr(k, s[k])
    # <= 0 is feasible; infeasible trials are kept off study.best_trials
    trial.set_constraint("cap_rate", s["capped"] - MAX_CAP_RATE)
    # one sample cover per case, for eyeballing naturalness in the dashboard
    trial.set_user_attr("samples", {row["case"]: row["covers"]
                                    for row in s["rows"] if row["repeat"] == 0})
    return (s["bpt"], s["ppl"]) if MULTI_OBJECTIVE else s["bpt"]


def load_models(model_path, codec_path, gpu_layers):
    print(f"Loading {model_path} + {codec_path} (n_gpu_layers={gpu_layers}) …")
    model = LlamaCppModel(model_path, n_ctx=8192, n_gpu_layers=gpu_layers)
    codec_model = LlamaCppModel(codec_path, n_ctx=2048, n_gpu_layers=gpu_layers)
    codec = LLMTextCodec(codec_model, temperature=CODEC_TEMP)

    # --- compression cache: the codec params are fixed across trials, so the
    #     secret -> bits pass is identical every time. Compute it once. ---
    original_encode_bits = codec.encode_bits
    cache: dict[str, tuple] = {}

    def cached_encode_bits(text: str):
        if text not in cache:
            cache[text] = original_encode_bits(text)
        return cache[text]

    codec.encode_bits = cached_encode_bits
    return model, codec


def run_baselines(model, codec):
    results = [(p, evaluate(p, model, codec, label="Baseline")) for p in BASELINES]
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)
    for p, s in results:
        print(f"  T={p['stego_temp']:.2f} topk={p['syncpool_topk']} minp={p['syncpool_minp']:.3f}  |  "
              f"bits/tok={s['bpt']:.3f}  ppl={s['ppl']:.2f}  tok={s['tokens']:.0f}  "
              f"msgs={s['msgs']:.2f}  capped={s['capped'] * 100:.0f}%  emoji={s['emoji'] * 100:.1f}%")
        for row in s["rows"]:
            if row["repeat"] == 0:
                print(f"      {row['case']}: {row['covers']}")


def run_study(model, codec, n_trials):
    study = optuna.create_study(
        study_name=STUDY_NAME if MULTI_OBJECTIVE else STUDY_NAME + "_bpt",
        storage=f"sqlite:///{DB_NAME}",
        directions=["maximize", "minimize"] if MULTI_OBJECTIVE else ["maximize"],
        load_if_exists=True,
        sampler=TPESampler(seed=42, multivariate=True),
    )

    print(f"\n[Storage] {DB_NAME}   [Dashboard] optuna-dashboard sqlite:///{DB_NAME}")
    print("Starting optimization...\n")
    study.optimize(lambda t: objective(t, model, codec), n_trials=n_trials, n_jobs=1)

    print("\n" + "=" * 60)
    print("OPTIMIZATION FINISHED")
    print("=" * 60)

    if MULTI_OBJECTIVE:
        print(f"\nPareto-optimal configs with cap-hit rate <= {MAX_CAP_RATE * 100:.0f}% "
              "(bits/token higher is better, ppl lower is better):")
        front = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
        for t in front:
            bpt, ppl = t.values
            p, u = t.params, t.user_attrs
            print(f"  bits/tok={bpt:5.3f}  ppl={ppl:6.2f}  tok={u.get('tokens', 0):5.1f}  "
                  f"msgs={u.get('msgs', 0):.2f}  capped={u.get('capped', 0) * 100:3.0f}%  "
                  f"emoji={u.get('emoji', 0) * 100:4.1f}%  |  "
                  f"stego_temp={p['stego_temp']:.2f}  syncpool_minp={p['syncpool_minp']:.3f}")
        print("\nPick the knee of this front: the point past which more "
              "bits/token costs a sharp jump in perplexity.")
    else:
        best = study.best_trial
        print(f"\nBest (Trial {best.number}): bits/token = {best.value:.3f}")
        for k, v in best.params.items():
            print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tune stego_temp / syncpool_minp with Optuna.")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--max-cap-rate", type=float, default=MAX_CAP_RATE,
                    help="max fraction of runs allowed to hit the message cap")
    ap.add_argument("--baseline", action="store_true",
                    help="score the fixed BASELINES configs first (old top-k settings)")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--codec-model", default=CODEC_MODEL_PATH)
    ap.add_argument("--gpu-layers", type=int, default=0, help="-1 = full GPU offload")
    args = ap.parse_args()
    REPEATS = args.repeats
    MAX_CAP_RATE = args.max_cap_rate

    model, codec = load_models(args.model, args.codec_model, args.gpu_layers)
    if args.baseline or args.baseline_only:
        run_baselines(model, codec)
    if not args.baseline_only:
        run_study(model, codec, args.trials)
