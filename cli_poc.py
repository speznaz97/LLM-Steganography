from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from stego import generate_stego, extract_stego

MODEL_PATH = "LFM2-8B-A1B-Q6_K.gguf"

if __name__ == "__main__":
    cfg = StegoConfig()

    print(f"Loading {MODEL_PATH} …")
    model = LlamaCppModel(
        MODEL_PATH,
        n_ctx=8192,
        #n_gpu_layers=-1,       # full GPU offload
        n_gpu_layers=0,      # CPU-only
    )
    codec = LLMTextCodec(model, temperature=1.0)

    messages =[
        {"role": "system",
         "content": "You are a coworker chatting on Slack. Write a natural, conversational response. Keep it to one paragraph."},
        {"role": "user",
         "content": "Hey, did you review the Q3 report?"}
    ]
    
    secret = "A secret message that is inside of a plain text omg"

    cover = generate_stego(messages, secret, model, codec, cfg)
    print(f"\n  Cover: {cover}\n")

    recovered = extract_stego(messages, cover, model, codec, cfg)
    ok = recovered.strip() == secret.strip()
    print(f"\n  Sent:      '{secret}'")
    print(f"  Recovered: '{recovered}'")
    print(f"  {'✅ OK' if ok else '❌ FAIL'}")