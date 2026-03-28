import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TTS_PYTHON = Path(os.getenv("AMADEUS_TTS_PYTHON", "/tmp/xtts_env/bin/python"))
DEFAULT_TTS_HOME = Path(os.getenv("AMADEUS_TTS_HOME", "/tmp/tts_home"))
DEFAULT_TTS_MPLCONFIGDIR = Path(os.getenv("AMADEUS_TTS_MPLCONFIGDIR", "/tmp/mplconfig"))
DEFAULT_TTS_OUTPUT_DIR = Path(os.getenv("AMADEUS_TTS_OUTPUT_DIR", tempfile.gettempdir())) / "amadeus_tts"
DEFAULT_TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_TTS_SPEAKER = os.getenv("AMADEUS_TTS_SPEAKER", "Uta Obando")
DEFAULT_TTS_LANGUAGE = os.getenv("AMADEUS_TTS_LANGUAGE", "en")
DEFAULT_TTS_DEVICE = os.getenv("AMADEUS_TTS_DEVICE", "auto")


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
    ) -> None:
        self.python_executable = Path(python_executable)
        self.tts_home = Path(tts_home)
        self.mplconfigdir = Path(mplconfigdir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.device = device

    def synthesize(self, text: str, speaker: str | None = None) -> dict[str, str]:
        if not text.strip():
            raise ValueError("Text is required for speech synthesis.")
        if not self.python_executable.exists():
            raise FileNotFoundError(f"TTS Python executable does not exist: {self.python_executable}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mplconfigdir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{uuid4().hex}.wav"
        resolved_speaker = speaker or self.speaker

        env = os.environ.copy()
        env["COQUI_TOS_AGREED"] = "1"
        env["TTS_HOME"] = str(self.tts_home)
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
