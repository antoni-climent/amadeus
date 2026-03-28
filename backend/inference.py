import json
import os
from pathlib import Path
import re
import sysconfig
from threading import Thread
import warnings

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "qwen3.5-4b-kurisu-sg-corpus_v4" / "checkpoint-300"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.txt"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_LOAD_IN_4BIT = True
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
UNFINISHED_THINK_PATTERN = re.compile(r"<think>[\s\S]*$")


def suppress_known_runtime_warnings() -> None:
    for module in (r"bitsandbytes\._ops", r"bitsandbytes\.backends\.cuda\.ops"):
        warnings.filterwarnings(
            "ignore",
            message=r".*_check_is_size will be removed in a future PyTorch release.*",
            category=FutureWarning,
            module=module,
        )


def python_dev_headers_available() -> bool:
    include_dir = sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_paths().get("include")
    if not include_dir:
        return False
    return Path(include_dir, "Python.h").exists()


def configure_unsloth_runtime() -> None:
    suppress_known_runtime_warnings()
    if python_dev_headers_available():
        return

    # Triton builds a small C extension against the active interpreter.
    # When Python dev headers are missing, force Unsloth onto the non-compiled path.
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


class ModelService:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    ) -> None:
        self.model_path = Path(model_path)
        self.system_prompt_path = Path(system_prompt_path)
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load_model(
        self,
        model_path: str | Path | None = None,
        max_seq_length: int | None = None,
        load_in_4bit: bool | None = None,
    ) -> dict[str, object]:
        configure_unsloth_runtime()
        try:
            from unsloth import FastVisionModel
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Unsloth in the current environment. "
                "Check that .amadeus_env has compatible versions of unsloth, transformers, torch, and GPU support. "
                f"Original error: {exc}"
            ) from exc

        resolved_model_path = Path(model_path).expanduser() if model_path else self.model_path
        resolved_max_seq_length = max_seq_length or self.max_seq_length
        resolved_load_in_4bit = self.load_in_4bit if load_in_4bit is None else load_in_4bit

        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {resolved_model_path}")

        if not self.is_loaded() or resolved_model_path != self.model_path:
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=str(resolved_model_path),
                max_seq_length=resolved_max_seq_length,
                dtype=None,
                load_in_4bit=resolved_load_in_4bit,
            )
            FastVisionModel.for_inference(self.model)

        self.model_path = resolved_model_path
        self.max_seq_length = resolved_max_seq_length
        self.load_in_4bit = resolved_load_in_4bit

        return {
            "loaded": True,
            "model_path": str(self.model_path),
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
        }

    def get_system_prompt(self) -> str:
        if not self.system_prompt_path.exists():
            return ""
        return self.system_prompt_path.read_text(encoding="utf-8").strip()

    def build_messages(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
        prompt = system_prompt if system_prompt is not None else self.get_system_prompt()
        history_messages = list(history or [])
        messages: list[dict[str, str]] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.extend(history_messages)

        user_message = {"role": "user", "content": message}
        messages.append(user_message)
        return messages, history_messages, user_message

    def prepare_inputs(self, messages: list[dict[str, str]]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.tokenizer(text=[text], return_tensors="pt").to(device)

    def clean_response(self, response: str) -> str:
        cleaned = THINK_BLOCK_PATTERN.sub("", response)
        cleaned = UNFINISHED_THINK_PATTERN.sub("", cleaned)
        return cleaned.strip()

    def build_history(
        self,
        history_messages: list[dict[str, str]],
        user_message: dict[str, str],
        response: str,
    ) -> list[dict[str, str]]:
        return history_messages + [user_message, {"role": "assistant", "content": response}]

    def generate(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        system_prompt: str | None = None,
    ) -> dict[str, object]:
        if not self.is_loaded():
            self.load_model()

        messages, history_messages, user_message = self.build_messages(
            message=message,
            history=history,
            system_prompt=system_prompt,
        )
        inputs = self.prepare_inputs(messages)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        response_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response = self.clean_response(response)

        return {
            "response": response,
            "history": self.build_history(history_messages, user_message, response),
            "model_loaded": True,
        }

    def stream_generate(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        system_prompt: str | None = None,
    ):
        from transformers import TextIteratorStreamer

        if not self.is_loaded():
            self.load_model()

        messages, history_messages, user_message = self.build_messages(
            message=message,
            history=history,
            system_prompt=system_prompt,
        )
        inputs = self.prepare_inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        raw_response = ""
        emitted_response = ""

        for chunk in streamer:
            raw_response += chunk
            cleaned_response = self.clean_response(raw_response)
            if len(cleaned_response) <= len(emitted_response):
                continue

            delta = cleaned_response[len(emitted_response):]
            emitted_response = cleaned_response
            yield self.format_sse_event("delta", {"delta": delta, "response": emitted_response})

        thread.join()
        final_response = self.clean_response(raw_response)
        yield self.format_sse_event(
            "done",
            {
                "response": final_response,
                "history": self.build_history(history_messages, user_message, final_response),
                "model_loaded": True,
            },
        )

    def format_sse_event(self, event: str, payload: dict[str, object]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


service = ModelService()


def run_cli() -> None:
    service.load_model()
    history: list[dict[str, str]] = []
    print("--- Conversation Started (type 'exit' or 'quit' to stop) ---")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = service.generate(message=user_input, history=history)
        print(f"Assistant: {result['response']}")
        history = result["history"]


if __name__ == "__main__":
    run_cli()
