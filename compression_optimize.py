import struct
import numpy as np
import time

from llm import LlamaCppModel
from codec import LLMTextCodec

# Using the model path from your previous trial runs
MODEL_PATH = "LFM2.5-8B-A1B-UD-Q6_K.gguf"

# Standardized test inputs (isolated to just secrets for Stage 1)
EVAL_SUITE = [
    # --- ORIGINAL SAMPLES ---
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
    },

    # --- NEW ENGLISH SAMPLES ---
    {
        "name": "Urgent Short (Signal)",
        "messages": [
            {"role": "system", "content": "You are a family member sending a quick, brief update on Signal. Keep it conversational and very short."},
            {"role": "user", "content": "Did you make it through the storm okay?"}
        ],
        "secret": "I am safe."
    },
    {
        "name": "Corporate Formal (Outlook)",
        "messages": [
            {"role": "system", "content": "You are a project manager replying to an executive. Maintain a professional, polite, and corporate tone. One paragraph max."},
            {"role": "user", "content": "Do we have the updated timeline for the client kickoff?"}
        ],
        "secret": "The launch date is pushed back by two weeks. Adjust plans accordingly."
    },

    # --- NEW RUSSIAN SAMPLES (CYRILLIC) ---
    {
        "name": "Casual Cyrillic (Telegram)",
        "messages": [
            {"role": "system", "content": "Вы — близкий друг, общающийся в Telegram. Пишите на естественном, разговорном русском языке. Будьте кратки."},
            {"role": "user", "content": "Привет! Пойдём сегодня вечером в кино?"}
        ],
        "secret": "Да, давай встретимся в семь."
    },
    {
        "name": "Work Cyrillic (Slack)",
        "messages": [
            {"role": "system", "content": "Вы — вежливый коллега по работе в Slack. Отвечайте лаконично, профессионально и по делу (один абзац)."},
            {"role": "user", "content": "Вы успели посмотреть вчерашний отчет по продажам?"}
        ],
        "secret": "Документы лежат в третьей папке на общем диске."
    },
    {
        "name": "Long Narrative Cyrillic (Catching Up)",
        "messages": [
            {"role": "system", "content": "Вы переписываетесь со старым знакомым в мессенджере. Напишите теплый, дружеский и развернутый ответ о ваших планах на выходные."},
            {"role": "user", "content": "Давно не виделись! Как планируешь провести эти выходные?"}
        ],
        "secret": "Встреча переносится на завтра, пароль прежний. Не выходи на связь до полудня."
    }
]

if __name__ == "__main__":
    print(f"Loading {MODEL_PATH} for Phase 1 (Codec Temperature Optimization)...")
    
    # We only need a small context size for compression to save memory/startup time
    model = LlamaCppModel(
        MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=0, 
    )

    print("\nStarting 1D sweep for optimal codec temperature...")
    # Dynamically build a table header to match the length of EVAL_SUITE
    headers = ["Temp", "Total"] + [f"C{i}" for i in range(len(EVAL_SUITE))]
    header_str = f"{headers[0]:<6} | {headers[1]:<6} | " + " | ".join(f"{h:<5}" for h in headers[2:])
    
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    best_temp = None
    min_total_bits = float('inf')

    # Sweep from 0.5 to 1.5 in fine increments
    temps = np.arange(0.5, 1.5, 0.01)
    
    for temp in temps:
        temp = round(float(temp), 2)
        
        case_bits = []
        total_bits = 0
        
        for case in EVAL_SUITE:
            codec = LLMTextCodec(model, temperature=temp, primer_text="Message:\n")
            
            # Encode and accumulate stats
            _, stats = codec.encode(case["secret"])
            bits_count = stats["bits"]
            case_bits.append(bits_count)
            total_bits += bits_count
        
        # Print row matching the dynamic table structure
        print(f"{temp:<6.2f} | {total_bits:<6} | " + " | ".join(f"{b:<5}" for b in case_bits))
        
        if total_bits < min_total_bits:
            min_total_bits = total_bits
            best_temp = temp

    print("-" * len(header_str))
    print(f"\nOptimal Codec Temperature: {best_temp:.2f}")
    print(f"Minimum Combined Payload: {min_total_bits} bits")
    print("-" * len(header_str))