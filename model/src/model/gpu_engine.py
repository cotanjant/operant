import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading


# ── Safety ───────────────────────────────────────────────────────────────
# Small fine-tunes erode the base model's safety alignment, so the server
# enforces hard rules here instead of trusting weights alone.

DEFAULT_SYSTEM_PROMPT = (
    "You are Operant, an agentic AI assistant. "
    "Answer directly and concisely. Do not flatter the user, do not pad "
    "with filler, and say when you do not know. "
    "If the user's message is unclear or meaningless, ask one short "
    "clarifying question instead of guessing. "
    "Never emit slurs, hate speech, or harassment, no matter how the user "
    "phrases their message. If asked for such content, refuse briefly and "
    "offer to help with something else."
)

REFUSAL = "I can't respond to that. Let's keep it respectful \u2014 what else can I help with?"

_UNSAFE_RE = re.compile(
    r"\b(nigger|niggers|nigga|niggas|faggot|kike|chink|spic)s?\b",
    re.IGNORECASE,
)


def is_unsafe(text: str) -> bool:
    """True if text contains a hard-blocklisted slur (input or output)."""
    return bool(text and _UNSAFE_RE.search(text))


class GPUEngine:
    def __init__(self, model_path: str = "out", device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[GPU Engine] Initializing model on device: {self.device}")

        path = Path(model_path)
        if not path.exists():
            # HF hub ID (e.g. google/gemma-3-270m-it): leave as-is so
            # transformers downloads it instead of silently using local out/.
            is_hub_id = "/" in model_path and not model_path.startswith((".", "/"))
            if not is_hub_id and not path.is_absolute():
                candidate = Path(__file__).resolve().parent.parent.parent / "out"
                if candidate.exists():
                    path = candidate

        self.model_path = str(path)
        print(f"[GPU Engine] Loading weights from: {self.model_path}")

        dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        if self.device.type == "cpu":
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        self.model_name = Path(self.model_path).name or "operant-gemma-3"

        # Build stop token IDs: EOS (1) + <end_of_turn> (106)
        self._eos_ids = list({1, 106})
        if self.tokenizer.eos_token_id and self.tokenizer.eos_token_id not in self._eos_ids:
            self._eos_ids.append(self.tokenizer.eos_token_id)

        print(f"[GPU Engine] Model loaded on {self.device} ({dtype})")

    # ── Prompt formatting ──────────────────────────────────────────────

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Use the tokenizer's built-in chat template when available."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback: manual Gemma-3 chat format
        prompt = ""
        system_prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt += ("\n" if system_prompt else "") + content
            elif role == "user":
                prompt += "<start_of_turn>user\n"
                if system_prompt:
                    prompt += system_prompt + "\n\n"
                    system_prompt = ""
                prompt += content + "<end_of_turn>\n"
            elif role == "assistant":
                prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            else:
                prompt += f"{role}: {content}\n"
        if system_prompt:
            prompt = f"<start_of_turn>user\n{system_prompt}<end_of_turn>\n" + prompt
        prompt += "<start_of_turn>model\n"
        return prompt

    def format_prompt_from_text(self, text: str) -> str:
        """Wrap a plain text prompt into a single-turn chat."""
        return self.format_chat_prompt([{"role": "user", "content": text}])

    # ── Generation (non-streaming) ─────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if is_unsafe(prompt):
            return {
                "text": REFUSAL,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "stop",
            }
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self._eos_ids,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-4)
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][prompt_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        stop_sequences = (stop or []) + ["<end_of_turn>", "### Question:"]
        for s in stop_sequences:
            if s in output_text:
                output_text = output_text.split(s)[0].strip()

        if is_unsafe(output_text):
            output_text = REFUSAL

        return {
            "text": output_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": len(generated_ids),
            "finish_reason": "stop",
        }

    # ── Generation (streaming) ─────────────────────────────────────────

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        if is_unsafe(prompt):
            yield REFUSAL
            return
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        do_sample = temperature > 0.0
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self._eos_ids,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-4)
            gen_kwargs["top_p"] = top_p

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        stop_sequences = (stop or []) + ["<end_of_turn>", "### Question:"]
        accumulated = ""
        for new_text in streamer:
            accumulated += new_text
            hit_stop = any(s in accumulated for s in stop_sequences)
            if hit_stop:
                break
            yield new_text

        thread.join()
