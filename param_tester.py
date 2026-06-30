import os
import time
import hashlib
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import numpy as np

# Core Imports from your project
from config import StegoConfig
from llm import LlamaCppModel
from codec import LLMTextCodec
from stego import generate_stego, extract_stego

# Default paths
DEFAULT_MODEL_PATH = "Qwen3.5-4B-Q6_K.gguf"
DEFAULT_CODEC_PATH = "Qwen3.5-0.8B-Q8_0.gguf"

SCENARIOS = {
    "Casual Short (WhatsApp)": {
        "system": "You are a close friend chatting on WhatsApp. Use casual English, stay brief, and be friendly. Do not use emojis.",
        "user": "Hey! Are we still on for coffee later?",
        "secret": "Yes, 5pm works."
    },
    "Planning Medium (Telegram)": {
        "system": "You are a helpful friend on Telegram. Answer in one natural paragraph. No formal greetings. Do not use emojis.",
        "user": "I'm looking for a good place to grab dinner tonight. Any suggestions?",
        "secret": "The package is hidden behind the loose brick in the garden wall."
    },
    "Work Informal (Slack)": {
        "system": "You are a coworker on Slack. Be professional but concise. Use one or two sentences maximum. Do not use emojis.",
        "user": "Did you get a chance to look at that spreadsheet I sent over this morning?",
        "secret": "Use the secondary encryption key: 8842-Alpha-Niner-X."
    },
    "Long Narrative (Catching Up)": {
        "system": "You are catching up with an old friend via messenger. Write a warm, natural paragraph about a weekend trip. Do not use emojis.",
        "user": "It's been forever! How was your trip to the mountains?",
        "secret": "The meeting is compromised. Do not go to the safehouse. Proceed directly to the airport and wait for the courier near Gate B7."
    },
    "Custom": {
        "system": "",
        "user": "",
        "secret": ""
    }
}

class StegoInteractiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("StegoChat Parameters & Visualizer Sandbox")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1e1e1e")

        # Models State
        self.model = None
        self.codec = None
        self.is_loading = False
        self.is_generating = False

        self._setup_style()
        self._build_ui()
        
        # Load first scenario by default
        self.scenario_var.set("Casual Short (WhatsApp)")
        self._apply_preset()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Color definitions (VSCode dark theme inspired)
        style.configure(".", background="#1e1e1e", foreground="#ffffff", fieldbackground="#333333")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabelframe", background="#1e1e1e", foreground="#ffffff", bordercolor="#3c3c3c")
        style.configure("TLabelframe.Label", background="#1e1e1e", foreground="#4caf50", font=("Helvetica", 10, "bold"))
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Helvetica", 10))
        style.configure("TButton", background="#333333", foreground="#ffffff", borderwidth=1, font=("Helvetica", 9, "bold"))
        style.map("TButton", background=[("active", "#424242")], foreground=[("active", "#ffffff")])
        style.configure("Action.TButton", background="#2e7d32", foreground="#ffffff", font=("Helvetica", 10, "bold"))
        style.map("Action.TButton", background=[("active", "#1b5e20")])

    def _build_ui(self):
        # Top-level paned container split left/right
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ==================== LEFT COLUMN (CONTROLS) ====================
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)

        # 1. Model Selection Box
        model_box = ttk.LabelFrame(left_frame, text=" 1. Models & Initialization ")
        model_box.pack(fill=tk.X, pady=5)

        ttk.Label(model_box, text="Main Model (Cover):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.main_model_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        ttk.Entry(model_box, textvariable=self.main_model_var, width=32).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(model_box, text="Browse", command=self._browse_main_model).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(model_box, text="Codec Model (Payload):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.codec_model_var = tk.StringVar(value=DEFAULT_CODEC_PATH)
        ttk.Entry(model_box, textvariable=self.codec_model_var, width=32).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(model_box, text="Browse", command=self._browse_codec_model).grid(row=1, column=2, padx=5, pady=2)

        self.btn_load = ttk.Button(model_box, text="Initialize Models", command=self._initialize_models_trigger)
        self.btn_load.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=6)

        # 2. Hyperparameter Sliders
        params_box = ttk.LabelFrame(left_frame, text=" 2. Hyperparameters ")
        params_box.pack(fill=tk.X, pady=5)

        # Stego Temperature
        ttk.Label(params_box, text="Stego Temp:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.temp_var = tk.DoubleVar(value=1.30)
        self.lbl_temp_val = ttk.Label(params_box, text="1.30")
        self.lbl_temp_val.grid(row=0, column=2, padx=5)
        scale_temp = ttk.Scale(params_box, from_=0.5, to=4.0, variable=self.temp_var, 
                               command=lambda v: self.lbl_temp_val.config(text=f"{float(v):.2f}"))
        scale_temp.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # SyncPool Top-K
        ttk.Label(params_box, text="SyncPool Top-K:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.top_k_var = tk.IntVar(value=35)
        self.lbl_topk_val = ttk.Label(params_box, text="35")
        self.lbl_topk_val.grid(row=1, column=2, padx=5)
        scale_topk = ttk.Scale(params_box, from_=5, to=256, variable=self.top_k_var,
                               command=lambda v: self.lbl_topk_val.config(text=f"{int(float(v))}"))
        scale_topk.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        params_box.columnconfigure(1, weight=1)

        # 3. Context & Dialog Scenarios
        context_box = ttk.LabelFrame(left_frame, text=" 3. Chat Context & Secret Message ")
        context_box.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Label(context_box, text="Preset Scenario:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.scenario_var = tk.StringVar()
        combo_scen = ttk.Combobox(context_box, textvariable=self.scenario_var, values=list(SCENARIOS.keys()), state="readonly")
        combo_scen.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        combo_scen.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())

        # System Prompt Frame
        ttk.Label(context_box, text="System Prompt:").grid(row=1, column=0, sticky="nw", padx=5, pady=2)
        self.system_text = scrolledtext.ScrolledText(context_box, height=3, bg="#2d2d2d", fg="white", insertbackground="white", font=("Helvetica", 9))
        self.system_text.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        self.system_text.bind("<Key>", lambda e: self._set_custom_preset())

        # User Input Frame
        ttk.Label(context_box, text="User Dialogue:").grid(row=2, column=0, sticky="nw", padx=5, pady=2)
        self.user_text = scrolledtext.ScrolledText(context_box, height=2, bg="#2d2d2d", fg="white", insertbackground="white", font=("Helvetica", 9))
        self.user_text.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        self.user_text.bind("<Key>", lambda e: self._set_custom_preset())

        # Secret Message Input
        ttk.Label(context_box, text="Secret Payload:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.secret_entry = ttk.Entry(context_box, width=35)
        self.secret_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.secret_entry.bind("<Key>", lambda e: self._set_custom_preset())
        context_box.columnconfigure(1, weight=1)

        # Action Buttons
        self.btn_run = ttk.Button(left_frame, text="Generate Cover & Validate Extraction", 
                                  style="Action.TButton", state=tk.DISABLED, command=self._run_stego_task)
        self.btn_run.pack(fill=tk.X, pady=5)

        # ==================== RIGHT COLUMN (OUTPUTS) ====================
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        # Output text displaying the generated Stego Cover
        output_box = ttk.LabelFrame(right_frame, text=" Generated Stegotext (Cover) ")
        output_box.pack(fill=tk.BOTH, expand=True, pady=5)
        self.cover_display = scrolledtext.ScrolledText(
            output_box, wrap=tk.WORD, state='disabled', bg="#181818", fg="#e0e0e0", 
            insertbackground="white", font=("Helvetica", 11), padx=10, pady=10
        )
        self.cover_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Metrics displaying extraction results
        metrics_box = ttk.LabelFrame(right_frame, text=" Execution Diagnostics ")
        metrics_box.pack(fill=tk.X, pady=5)

        # Metric Labels
        self.lbl_verify = tk.Label(metrics_box, text="Validation:  Pending Loading", fg="#ffeb3b", bg="#1e1e1e", font=("Helvetica", 11, "bold"))
        self.lbl_verify.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        self.lbl_extracted = tk.Label(metrics_box, text="Extracted Payload: N/A", fg="#e0e0e0", bg="#1e1e1e", font=("Courier", 10, "italic"))
        self.lbl_extracted.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        self.lbl_bpt = ttk.Label(metrics_box, text="Embedding Capacity: - bpt")
        self.lbl_bpt.grid(row=2, column=0, sticky="w", padx=10, pady=3)

        self.lbl_bits = ttk.Label(metrics_box, text="Payload Bits: - bits")
        self.lbl_bits.grid(row=3, column=0, sticky="w", padx=10, pady=3)

        self.lbl_tokens = ttk.Label(metrics_box, text="Cover Size: - tokens")
        self.lbl_tokens.grid(row=2, column=1, sticky="w", padx=10, pady=3)

        self.lbl_time = ttk.Label(metrics_box, text="Generation Time: - s")
        self.lbl_time.grid(row=3, column=1, sticky="w", padx=10, pady=3)
        metrics_box.columnconfigure(0, weight=1)
        metrics_box.columnconfigure(1, weight=1)

        # Status Bar
        self.status_bar = tk.Label(self.root, text="Ready. Load LLMs to begin.", bd=1, relief=tk.SUNKEN, anchor="w", bg="#252526", fg="#9e9e9e", font=("Helvetica", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # File browsing controls
    def _browse_main_model(self):
        filename = filedialog.askopenfilename(filetypes=[("GGUF files", "*.gguf")])
        if filename: self.main_model_var.set(filename)

    def _browse_codec_model(self):
        filename = filedialog.askopenfilename(filetypes=[("GGUF files", "*.gguf")])
        if filename: self.codec_model_var.set(filename)

    # Preset application
    def _apply_preset(self):
        pname = self.scenario_var.get()
        if pname in SCENARIOS:
            preset = SCENARIOS[pname]
            
            self.system_text.delete("1.0", tk.END)
            self.system_text.insert("1.0", preset["system"])
            
            self.user_text.delete("1.0", tk.END)
            self.user_text.insert("1.0", preset["user"])
            
            self.secret_entry.delete(0, tk.END)
            self.secret_entry.insert(0, preset["secret"])

    def _set_custom_preset(self):
        if self.scenario_var.get() != "Custom":
            self.scenario_var.set("Custom")

    # Thread-safe logging status
    def _log_status(self, text):
        self.root.after(0, lambda: self.status_bar.config(text=text))

    # Trigger initialization thread
    def _initialize_models_trigger(self):
        if self.is_loading: return
        self.is_loading = True
        self.btn_load.config(state=tk.DISABLED, text="Initializing Models (Blocking UI Thread)...")
        self._log_status("Spinning up local background thread to load weights...")
        
        threading.Thread(target=self._load_models_worker, daemon=True).start()

    def _load_models_worker(self):
        m_path = self.main_model_var.get().strip()
        c_path = self.codec_model_var.get().strip()

        if not os.path.exists(m_path):
            self._log_status(f"Error: main model path '{m_path}' not found.")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Could not find GGUF model: {m_path}"))
            self.is_loading = False
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL, text="Initialize Models"))
            return

        if not os.path.exists(c_path):
            self._log_status(f"Error: codec model path '{c_path}' not found.")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Could not find GGUF model: {c_path}"))
            self.is_loading = False
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL, text="Initialize Models"))
            return

        try:
            self._log_status("Loading main LLM model into CPU/GPU cache...")
            model = LlamaCppModel(m_path, n_ctx=8192, n_gpu_layers=0)
            
            self._log_status("Loading sub-codec LLM model into cache...")
            codec_model = LlamaCppModel(c_path, n_ctx=2048, n_gpu_layers=0)
            codec = LLMTextCodec(codec_model, temperature=1.12)

            self.model = model
            self.codec = codec
            
            self._log_status("Initialization successful! UI enabled.")
            self.root.after(0, self._on_models_loaded)
        except Exception as e:
            self._log_status(f"Initialization failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Crash during weights load", str(e)))
        finally:
            self.is_loading = False

    def _on_models_loaded(self):
        self.btn_load.config(text="Models Online ✓", state=tk.DISABLED)
        self.btn_run.config(state=tk.NORMAL)
        self.lbl_verify.config(text="Validation: Pending Generation", fg="#ffeb3b")

    # Steganography Processing Thread
    def _run_stego_task(self):
        if self.is_generating or not self.model or not self.codec: return
        
        self.is_generating = True
        self.btn_run.config(state=tk.DISABLED, text="Orchestrating Stegotext... ⏳")
        self._log_status("Running evaluation Suite pipeline...")
        
        # Pull parameters dynamically from GUI
        temp = float(self.temp_var.get())
        top_k = int(self.top_k_var.get())
        system = self.system_text.get("1.0", tk.END).strip()
        user = self.user_text.get("1.0", tk.END).strip()
        secret = self.secret_entry.get()

        if not secret:
            messagebox.showwarning("Incomplete Form", "Secret payload is empty.")
            self.btn_run.config(state=tk.NORMAL, text="Generate Cover & Validate Extraction")
            self.is_generating = False
            return

        # Prepare payload config package
        cfg = StegoConfig(
            stego_temp=temp,
            syncpool_topk=top_k,
            crypto_key="shared_secret_password_123",
            max_gen_tokens=512
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        # Trigger background run
        threading.Thread(target=self._stego_pipeline_worker, args=(messages, secret, cfg), daemon=True).start()

    def _stego_pipeline_worker(self, messages, secret, cfg):
        t_start = time.time()
        try:
            # 1. Embed & Generate
            self._log_status("Processing Stego Generation (SyncPool + Discop)...")
            cover = generate_stego(messages, secret, self.model, self.codec, cfg)
            t_gen = time.time() - t_start

            # 2. Extract & Validate
            self._log_status("Decoupling extraction test back via zero-header decoding...")
            t_ext_start = time.time()
            
            # Incorporating Option B's dynamic safe boundary guard:
            # First, estimate payload bits from codec
            raw_bits, _ = self.codec.encode_bits(secret)
            payload_bits = len(raw_bits)
            
            recovered = extract_stego(messages, cover, self.model, self.codec, cfg)
            t_ext = time.time() - t_ext_start

            cover_ids = self.model.tokenize(cover, add_bos=False, special=False)
            cover_tokens = max(1, len(cover_ids))
            bpt = payload_bits / cover_tokens

            # Direct state check
            is_valid = (recovered.strip() == secret.strip())
            
            self.root.after(0, lambda: self._update_results_ui(
                cover, recovered, is_valid, payload_bits, cover_tokens, bpt, t_gen, None
            ))

        except Exception as e:
            self._log_status(f"Fatal execution crash: {e}")
            self.root.after(0, lambda msg=str(e): self._update_results_ui_error(msg))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="Generate Cover & Validate Extraction"))
            self.is_generating = False

    def _update_results_ui(self, cover, recovered, is_valid, bits, tokens, bpt, elapsed, error=None):
        self.cover_display.config(state='normal')
        self.cover_display.delete("1.0", tk.END)
        self.cover_display.insert("1.0", cover)
        self.cover_display.config(state='disabled')

        if is_valid:
            self.lbl_verify.config(text="Validation: ✅ Match (Zero-Header Intact)", fg="#4caf50")
            self.lbl_extracted.config(text=f"Extracted: '{recovered}'", fg="#4caf50")
        else:
            self.lbl_verify.config(text="Validation: ❌ Mismatch (CRC failure or desync)", fg="#f44336")
            self.lbl_extracted.config(text=f"Extracted: '{recovered}'", fg="#f44336")

        self.lbl_bits.config(text=f"Payload Bits: {bits} bits")
        self.lbl_tokens.config(text=f"Cover Size: {tokens} tokens")
        self.lbl_bpt.config(text=f"Embedding Capacity: {bpt:.3f} bpt")
        self.lbl_time.config(text=f"Generation Time: {elapsed:.2f} s")
        self._log_status("Execution done.")

    def _update_results_ui_error(self, error_msg):
        self.cover_display.config(state='normal')
        self.cover_display.delete("1.0", tk.END)
        self.cover_display.insert("1.0", f"Error during execution pipeline:\n{error_msg}")
        self.cover_display.config(state='disabled')

        self.lbl_verify.config(text="Validation: ❌ Crash", fg="#f44336")
        self.lbl_extracted.config(text="Extracted: N/A", fg="#9e9e9e")
        self.lbl_bits.config(text="Payload Bits: -")
        self.lbl_tokens.config(text="Cover Size: -")
        self.lbl_bpt.config(text="Embedding Capacity: -")
        self.lbl_time.config(text="Generation Time: -")


if __name__ == "__main__":
    # Workaround window lag on high density displays
    root = tk.Tk()
    app = StegoInteractiveApp(root)
    root.mainloop()