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
from integrity import session_fingerprint, require_match

MODEL_PATH = "Qwen3.5-4B-Q6_K.gguf"#"LFM2-8B-A1B-Q6_K.gguf"
CODEC_MODEL_PATH = "Qwen3.5-0.8B-Q8_0.gguf"

def run_chat_client(name, send_queue, recv_queue):
    print(f"[{name}] Booting up isolated environment...")
    
    my_name = name
    their_name = "Bob" if name == "Alice" else "Alice"
    
    cfg = StegoConfig(
        stego_temp=1.38,
        syncpool_topk=35
    )

    # Load LLMs
    model = LlamaCppModel(MODEL_PATH, n_ctx=8192, n_gpu_layers=0)
    codec_model = LlamaCppModel(CODEC_MODEL_PATH, n_ctx=2048, n_gpu_layers=0)
    codec = LLMTextCodec(codec_model, temperature=1.12)
    
    my_fp = session_fingerprint(model, codec, cfg)
    send_queue.put(("__handshake__", my_fp))
    tag, peer_fp = recv_queue.get()                # blocks until peer sends theirs
    assert tag == "__handshake__"
    require_match(my_fp, peer_fp)                  # raises -> refuses to run on mismatch
    # --- TRANSCRIPT STATE ---
    chat_log = [
        "Bob: Hey! How's your week going? Hope you're doing well."
    ]
    
    def get_messages(speaker_name):
        """
        Builds a perfectly alternating User/Assistant chat history 
        where the last message is ALWAYS 'user' and the generating 
        speaker is ALWAYS 'assistant'.
        """
        other_name = "Bob" if speaker_name == "Alice" else "Alice"
        
        # Explicitly enforce the persona and forbid the "AI Assistant" signature
        system_prompt = (
            f"You are {speaker_name}, chatting with your close friend {other_name} on a messenger. "
            f"Write a natural, warm, and highly detailed response as {speaker_name}. "
            f"Share plenty of details about your week, your thoughts, and ask open-ended questions. "
            #f"Do not include any formal sign-offs, and absolutely never refer to yourself as an AI or Assistant."
        )
        formatted = [{"role": "system", "content": system_prompt}]
        
        # Build roles backwards so the most recent message is always 'user'
        roles = []
        is_user = True
        for _ in range(len(chat_log)):
            roles.append("user" if is_user else "assistant")
            is_user = not is_user
        
        roles.reverse()  # Restore forward chronological order
        
        for msg, role in zip(chat_log, roles):
            # Strip the sender prefix so the LLM doesn't see duplicate labels
            clean_content = msg
            if msg.startswith("Alice: "):
                clean_content = msg[7:]
            elif msg.startswith("Bob: "):
                clean_content = msg[5:]
                
            formatted.append({"role": role, "content": clean_content})
            
        return formatted
    
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

    chat_display.tag_config("you", foreground="#4caf50", font=("Helvetica", 11, "bold"))
    chat_display.tag_config("them", foreground="#2196f3", font=("Helvetica", 11, "bold"))
    chat_display.tag_config("cover", foreground="#e0e0e0") 
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
                # 1. Grab dynamically formatted messages for ME and Generate
                msgs = get_messages(my_name)
                cover = generate_stego(msgs, sec, model, codec, cfg)
                
                # 2. Append the generated text to our raw log
                chat_log.append(f"{my_name}: {cover}")
                
                # 3. Update UI & Send
                root.after(0, lambda: log_msg("You", cover, secret=sec))
                send_queue.put(cover)
            except Exception as e:
                err_msg = f"Generation error: {e}"
                root.after(0, lambda msg=err_msg: log_msg("System", msg, is_system=True))
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
                    # 1. Decode using THEIR perspective so the roles match their generation math
                    msgs = get_messages(their_name)
                    recovered = extract_stego(msgs, cover, model, codec, cfg)
                    root.after(0, lambda: log_msg("System", f"Extracted: {recovered}", is_system=True))
                    
                    # 2. Append their cover to our raw log
                    chat_log.append(f"{their_name}: {cover}")
                    
                except Exception as e:
                    err_msg = f"Extraction failed: {e}"
                    root.after(0, lambda msg=err_msg: log_msg("System", msg, is_system=True))
                finally:
                    root.after(0, lambda: set_status("Idle"))

            threading.Thread(target=decode_worker, daemon=True).start()
                
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