import argparse
from pathlib import Path

import soundfile as sf


DEFAULT_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEVICE = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument("--instruct", default="")
    return parser.parse_args()


def resolve_model_kwargs(device: str) -> dict[str, object]:
    import importlib.util
    import torch

    if device == "cpu":
        device_map = "cpu"
        dtype = torch.float32
    elif device == "cuda":
        device_map = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32

    kwargs: dict[str, object] = {
        "device_map": device_map,
        "dtype": dtype,
    }
    if device_map != "cpu" and importlib.util.find_spec("flash_attn") is not None:
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model_name,
        **resolve_model_kwargs(args.device),
    )
    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct or None,
    )
    sf.write(str(output_path), wavs[0], sample_rate)


if __name__ == "__main__":
    main()
