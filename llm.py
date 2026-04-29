import numpy as np
from llama_cpp import Llama, llama_get_logits_ith

class LlamaCppModel:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1, **kwargs):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
            **kwargs,
        )
        self._n_vocab = self.llm.n_vocab()
        self.eos_id   = self.llm.token_eos()
        self.bos_id   = self.llm.token_bos()

        self.special_ids: set[int] = set()
        for tid in (self.eos_id, self.bos_id):
            if tid is not None and tid >= 0:
                self.special_ids.add(tid)

        for marker in["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            try:
                ids = self.tokenize(marker, add_bos=False, special=True)
                self.special_ids.update(ids)
            except Exception:
                pass

        if hasattr(self.llm, '_ctx') and hasattr(self.llm._ctx, 'ctx'):
            self._raw_ctx = self.llm._ctx.ctx
        elif hasattr(self.llm, 'ctx'):
            self._raw_ctx = self.llm.ctx
        else:
            raise RuntimeError("Cannot find llama_context pointer")

        self._jinja_tpl = None
        raw = self.llm.metadata.get("tokenizer.chat_template")
        if raw:
            try:
                import jinja2
                env = jinja2.Environment(autoescape=False)
                env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
                self._jinja_tpl = env.from_string(raw)
            except ImportError:
                print("  ⚠ jinja2 not installed → ChatML fallback")

        self._verify_logits()

    def _verify_logits(self):
        self.reset()
        self.eval([self.bos_id])
        lg = self.get_logits()
        hi = np.abs(lg).max()
        if hi < 1e-6:
            raise RuntimeError("Logits are all zeros after eval(BOS). API change?")
        print(f"  [OK] logits verified: range[{lg.min():.1f}, {lg.max():.1f}]")
        self.reset()

    @property
    def n_vocab(self) -> int: return self._n_vocab
    @property
    def n_tokens(self) -> int: return self.llm.n_tokens

    def tokenize(self, text: str, add_bos: bool = False, special: bool = True) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=special)

    def detokenize(self, tokens: list[int], skip_special: bool = False) -> str:
        if skip_special:
            tokens =[t for t in tokens if t not in self.special_ids]
        return self.llm.detokenize(tokens).decode("utf-8", errors="replace")

    def reset(self): self.llm.reset()
    def eval(self, tokens: list[int]): self.llm.eval(tokens)

    def get_logits(self) -> np.ndarray:
        ptr = llama_get_logits_ith(self._raw_ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).copy()

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw) -> str:
        enable_thinking = kw.pop("enable_thinking", False)
        if self._jinja_tpl:
            try:
                return self._jinja_tpl.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    bos_token=(self.detokenize([self.bos_id]) if self.bos_id >= 0 else ""),
                    eos_token=(self.detokenize([self.eos_id]) if self.eos_id >= 0 else ""),
                    enable_thinking=enable_thinking,
                    **kw,
                )
            except Exception: pass
        
        parts =[f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
        if add_generation_prompt:
            if enable_thinking: parts.append("<|im_start|>assistant\n<think>\n")
            else: parts.append("<|im_start|>assistant\n<think>\n</think>\n\n")
        return "\n".join(parts)