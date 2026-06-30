from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from stego import generate_stego, extract_stego
from integrity import session_fingerprint, require_match

MODEL_PATH = "Qwen3.5-4B-Q6_K.gguf"#"LFM2-8B-A1B-Q6_K.gguf"
CODEC_MODEL_PATH = "Qwen3.5-0.8B-Q8_0.gguf"

if __name__ == "__main__":
    cfg = StegoConfig(
        stego_temp=1.22,
        syncpool_topk=47
    )

    print(f"Loading {MODEL_PATH} …")
    model = LlamaCppModel(
        MODEL_PATH,
        n_ctx=8192,
        #n_gpu_layers=-1,       # full GPU offload
        n_gpu_layers=0,      # CPU-only
    )
    codec_model = LlamaCppModel(
        CODEC_MODEL_PATH,
        n_ctx=2048,
        #n_gpu_layers=-1,       # full GPU offload
        n_gpu_layers=0,      # CPU-only
    )
    codec = LLMTextCodec(model, temperature=1.06)

    messages =[
        {"role": "system",
         "content": "You are a coworker chatting on Slack. Write a natural, conversational response. Keep it to one paragraph. No emojis"},
        {"role": "user",
         "content": "Hey, did you review the Q3 report?"}
    ]
    
    secret = "A secret message that is inside of a plain text omg"

    print("Running handshake self-test...")
    fp = session_fingerprint(model, codec, cfg)
    require_match(fp, fp)
    print(f"Handshake passed: {fp}")
    
    cover = generate_stego(messages, secret, model, codec, cfg)
    print(f"\n  Cover: {cover}\n")

    recovered = extract_stego(messages, cover, model, codec, cfg)
    ok = recovered.strip() == secret.strip()
    print(f"\n  Sent:      '{secret}'")
    print(f"  Recovered: '{recovered}'")
    print(f"  {'✅ OK' if ok else '❌ FAIL'}")