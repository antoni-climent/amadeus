import os
import subprocess
from pathlib import Path
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TTS_PYTHON = Path(os.getenv("AMADEUS_TTS_PYTHON", str(ROOT_DIR / ".tts_env" / "bin" / "python")))
DEFAULT_TTS_HOME = Path(os.getenv("AMADEUS_TTS_HOME", str(ROOT_DIR / ".runtime" / "tts_home")))
DEFAULT_TTS_MPLCONFIGDIR = Path(os.getenv("AMADEUS_TTS_MPLCONFIGDIR", str(ROOT_DIR / ".runtime" / "mplconfig")))
DEFAULT_TTS_OUTPUT_DIR = Path(os.getenv("AMADEUS_TTS_OUTPUT_DIR", str(ROOT_DIR / ".runtime" / "amadeus_tts")))
DEFAULT_TTS_MODEL_NAME = os.getenv("AMADEUS_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_TTS_SPEAKER = os.getenv("AMADEUS_TTS_SPEAKER", "Ono_anna")
DEFAULT_TTS_LANGUAGE = os.getenv("AMADEUS_TTS_LANGUAGE", "English")
DEFAULT_TTS_DEVICE = os.getenv("AMADEUS_TTS_DEVICE", "auto")
DEFAULT_TTS_INSTRUCT = os.getenv("AMADEUS_TTS_INSTRUCT", "")


class TtsService:
    def __init__(
        self,
        python_executable: Path = DEFAULT_TTS_PYTHON,
        tts_home: Path = DEFAULT_TTS_HOME,
        mplconfigdir: Path = DEFAULT_TTS_MPLCONFIGDIR,
        output_dir: Path = DEFAULT_TTS_OUTPUT_DIR,
        model_name: str = DEFAULT_TTS_MODEL_NAME,
        speaker: str = DEFAULT_TTS_SPEAKER,
        language: str = DEFAULT_TTS_LANGUAGE,
        device: str = DEFAULT_TTS_DEVICE,
        instruct: str = DEFAULT_TTS_INSTRUCT,
    ) -> None:
        self.python_executable = Path(python_executable)
        self.tts_home = Path(tts_home)
        self.mplconfigdir = Path(mplconfigdir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.device = device
        self.instruct = instruct

    def synthesize(self, text: str, speaker: str | None = None) -> dict[str, str]:
        if not text.strip():
            raise ValueError("Text is required for speech synthesis.")
        if not self.python_executable.exists():
            raise FileNotFoundError(f"TTS Python executable does not exist: {self.python_executable}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mplconfigdir.mkdir(parents=True, exist_ok=True)
        self.tts_home.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{uuid4().hex}.wav"
        resolved_speaker = speaker or self.speaker

        env = os.environ.copy()
        env["HF_HOME"] = str(self.tts_home)
        env["MPLCONFIGDIR"] = str(self.mplconfigdir)

        command = [
            str(self.python_executable),
            str(ROOT_DIR / "backend" / "tts_runner.py"),
            "--text",
            text,
            "--speaker",
            resolved_speaker,
            "--language",
            self.language,
            "--output",
            str(output_path),
            "--model-name",
            self.model_name,
            "--device",
            self.device,
        ]
        if self.instruct:
            command.extend(["--instruct", self.instruct])

        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or "Unknown TTS failure."
            raise RuntimeError(detail)

        return {
            "path": str(output_path),
            "speaker": resolved_speaker,
            "language": self.language,
        }


tts_service = TtsService()
