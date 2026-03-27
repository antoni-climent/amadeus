from pathlib import Path
import re

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "qwen3.5-4b-kurisu-sg-corpus_v4" / "checkpoint-300"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.txt"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_LOAD_IN_4BIT = True
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


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
        from unsloth import FastVisionModel

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

    def generate(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        system_prompt: str | None = None,
    ) -> dict[str, object]:
        if not self.is_loaded():
            self.load_model()

        prompt = system_prompt if system_prompt is not None else self.get_system_prompt()
        messages: list[dict[str, str]] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.extend(history or [])
        messages.append({"role": "user", "content": message})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer(text=[text], return_tensors="pt").to(device)

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
        response = THINK_BLOCK_PATTERN.sub("", response).strip()

        return {
            "response": response,
            "history": messages + [{"role": "assistant", "content": response}],
            "model_loaded": True,
        }


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
        history = [message for message in result["history"] if message.get("role") != "system"]


if __name__ == "__main__":
    run_cli()
