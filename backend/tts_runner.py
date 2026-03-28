import argparse
from pathlib import Path

from TTS.api import TTS
import torch


DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_DEVICE = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_gpu = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    model = TTS(args.model_name, gpu=use_gpu)
    model.tts_to_file(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        file_path=str(output_path),
    )


if __name__ == "__main__":
    main()
