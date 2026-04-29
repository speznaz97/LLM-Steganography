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
         "content": "Hey, did you review the Q3 report?"},
        {"role": "assistant",
         "content": "Yes, I looked at it this morning. The numbers look solid, but I think we need to adjust the marketing budget for next quarter."},
        {"role": "user",
         "content": "Agreed. I'll schedule a meeting with Sarah to discuss the changes. Are you free tomorrow at 2 PM?"}
    ]
    
    secret = "The backdoor in the production server is still active. I will extract the database at midnight."

    cover = generate_stego(messages, secret, model, codec, cfg)
    print(f"\n  Cover: {cover}\n")

    recovered = extract_stego(messages, cover, model, codec, cfg)
    ok = recovered.strip() == secret.strip()
    print(f"\n  Sent:      '{secret}'")
    print(f"  Recovered: '{recovered}'")
    print(f"  {'✅ OK' if ok else '❌ FAIL'}")