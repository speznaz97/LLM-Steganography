import multiprocessing as mp
import tkinter as tk
from tkinter import scrolledtext, font
import threading
import time

# Import the core logic from your existing script
from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from stego import generate_stego, extract_stego

MODEL_PATH = "LFM2-8B-A1B-Q6_K.gguf"

def run_chat_client(name, send_queue, recv_queue):
    """
    Runs an entirely isolated instance of the chat client.
    Has its own memory space, its own LLM instance, and its own context history.
    """
    print(f"[{name}] Booting up isolated environment...")
    
    # Apply the optimized Pareto configuration
    cfg = StegoConfig(
        stego_temp=1.4231,
        top_k=104,
        prob_threshold=0.0051,
        rep_penalty=1.1243,
        retoken_window=10,
        tail_max=30,  
        tail_min=1,
    )

    # Load LLM into this process's memory
    model = LlamaCppModel(MODEL_PATH, n_ctx=8192, n_gpu_layers=0)
    codec = LLMTextCodec(model, temperature=1.0)
    
    # Shared deterministic starting context. Both clients MUST start identically.
    # We append all sent/received messages as "user" so the LLM always answers as "assistant"
    messages =[
        {"role": "system", "content": "You are a close friend chatting on a messenger. Write natural, conversational responses. Keep it to one paragraph."}
    ]
    
    # --- GUI Setup ---
    root = tk.Tk()
    root.title(f"StegoChat: {name} (Idle)")
    root.geometry("600x600")
    root.configure(bg="#1e1e1e")

    custom_font = font.Font(family="Helvetica", size=11)
    
    chat_display = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, state='disabled', bg="#2d2d2d", fg="#ffffff",
        font=custom_font, padx=10, pady=10, insertbackground="white"
    )
    chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Tags for coloring UI text
    chat_display.tag_config("you", foreground="#4caf50", font=("Helvetica", 11, "bold"))
    chat_display.tag_config("them", foreground="#2196f3", font=("Helvetica", 11, "bold"))
    # The actual cover message text
    chat_display.tag_config("cover", foreground="#e0e0e0") 
    # The extracted secret payload
    chat_display.tag_config("secret", foreground="#ff5252", font=("Courier", 10, "italic"))
    chat_display.tag_config("system", foreground="#ffeb3b", font=("Helvetica", 10, "italic"))

    def set_status(status):
        root.title(f"StegoChat: {name} ({status})")

    def log_msg(sender, text, secret=None, is_system=False):
        chat_display.config(state='normal')
        if is_system:
             chat_display.insert(tk.END, f"⚙️ {text}\n\n", "system")
        else:
            tag = "you" if sender == "You" else "them"
            chat_display.insert(tk.END, f"[{sender}]\n", tag)
            chat_display.insert(tk.END, f"{text}\n", "cover")
            if secret:
                chat_display.insert(tk.END, f"↳ 🔓 Secret: {secret}\n", "secret")
            chat_display.insert(tk.END, "\n")
            
        chat_display.config(state='disabled')
        chat_display.yview(tk.END)

    # --- Input UI ---
    input_frame = tk.Frame(root, bg="#1e1e1e")
    input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

    tk.Label(input_frame, text="Secret Payload:", bg="#1e1e1e", fg="white").grid(row=0, column=0, sticky='w')
    secret_entry = tk.Entry(input_frame, width=50, bg="#424242", fg="white", insertbackground="white")
    secret_entry.grid(row=0, column=1, padx=5, pady=5)

    def send_stego():
        sec = secret_entry.get()
        if not sec: return
        secret_entry.delete(0, tk.END)
        
        def worker():
            root.after(0, lambda: set_status("Generating Cover..."))
            try:
                # 1. Generate Cover
                cover = generate_stego(messages, sec, model, codec, cfg)
                
                # 2. Update local context history
                messages.append({"role": "user", "content": cover})
                
                # 3. Show in UI
                root.after(0, lambda: log_msg("You", cover, secret=sec))
                
                # 4. SEND OVER "WIRE" (Simulated network. Only cover text is sent!)
                send_queue.put(cover)
            except Exception as e:
                root.after(0, lambda: log_msg("System", f"Generation error: {e}", is_system=True))
            finally:
                root.after(0, lambda: set_status("Idle"))
                
        threading.Thread(target=worker, daemon=True).start()

    tk.Button(input_frame, text="Send Secret", bg="#d32f2f", fg="white", 
              command=send_stego).grid(row=0, column=2, padx=5)

    # --- Receiver Loop ---
    def check_queue():
        while not recv_queue.empty():
            # RECEIVED OVER "WIRE". We only receive the plain cover text.
            cover = recv_queue.get() 
            log_msg("Them", cover) # Log the cover immediately
            
            def decode_worker():
                root.after(0, lambda: set_status("Extracting Secret..."))
                try:
                    # 1. Extract secret using context BEFORE the cover is appended
                    recovered = extract_stego(messages, cover, model, codec, cfg)
                    root.after(0, lambda: log_msg("System", f"Extracted: {recovered}", is_system=True))
                except Exception as e:
                    root.after(0, lambda: log_msg("System", f"Extraction failed: {e}", is_system=True))
                
                # 2. Append cover to local context so we stay synced for the next turn!
                messages.append({"role": "user", "content": cover})
                root.after(0, lambda: set_status("Idle"))

            threading.Thread(target=decode_worker, daemon=True).start()
                
        # Check network queue every 200ms
        root.after(200, check_queue)

    log_msg("System", "Chat started. Context synchronized. Ready.", is_system=True)
    root.after(200, check_queue)
    root.mainloop()


if __name__ == "__main__":
    # Required for Windows multiprocessing compatibility
    mp.freeze_support() 
    
    print("\n" + "="*60)
    print("STARTING DECENTRALIZED STEGO CHAT POC")
    print("Spawning two isolated processes. Please check RAM usage.")
    print("="*60 + "\n")

    # Create two directional Queues to act as our "Internet"
    # Process A writes to q_A_to_B. Process B reads from q_A_to_B.
    q_A_to_B = mp.Queue()
    q_B_to_A = mp.Queue()

    # Launch Alice (Process A)
    process_A = mp.Process(target=run_chat_client, args=("Alice", q_A_to_B, q_B_to_A))
    process_A.start()

    # Launch Bob (Process B)
    # Give Alice a second to grab GPU/RAM locks if needed before starting Bob
    time.sleep(2) 
    process_B = mp.Process(target=run_chat_client, args=("Bob", q_B_to_A, q_A_to_B))
    process_B.start()

    # Keep the main launcher script alive while the GUI windows are open
    process_A.join()
    process_B.join()
    
    print("Chat closed.")