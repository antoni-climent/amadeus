import json
import os
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TTS_OUTPUT_DIR = Path(os.getenv("AMADEUS_TTS_OUTPUT_DIR", str(ROOT_DIR / ".runtime" / "amadeus_tts")))
DEFAULT_TTS_SPEAKER = os.getenv("AMADEUS_TTS_SPEAKER", "Ono_anna")
DEFAULT_TTS_LANGUAGE = os.getenv("AMADEUS_TTS_LANGUAGE", "English")
DEFAULT_TTS_INSTRUCT = os.getenv("AMADEUS_TTS_INSTRUCT", "")
DEFAULT_TTS_URL = os.getenv("AMADEUS_TTS_URL", "http://127.0.0.1:8001")
DEFAULT_TTS_TIMEOUT_SECONDS = float(os.getenv("AMADEUS_TTS_TIMEOUT_SECONDS", "600"))


class TtsService:
    def __init__(
        self,
        output_dir: Path = DEFAULT_TTS_OUTPUT_DIR,
        speaker: str = DEFAULT_TTS_SPEAKER,
        language: str = DEFAULT_TTS_LANGUAGE,
        instruct: str = DEFAULT_TTS_INSTRUCT,
        base_url: str = DEFAULT_TTS_URL,
        timeout_seconds: float = DEFAULT_TTS_TIMEOUT_SECONDS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.speaker = speaker
        self.language = language
        self.instruct = instruct
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def synthesize(self, text: str, speaker: str | None = None) -> dict[str, str]:
        if not text.strip():
            raise ValueError("Text is required for speech synthesis.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{uuid4().hex}.wav"
        resolved_speaker = speaker or self.speaker

        payload = json.dumps(
            {
                "text": text,
                "speaker": resolved_speaker,
                "language": self.language,
                "instruct": self.instruct or None,
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            url=f"{self.base_url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                audio_bytes = response.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or f"TTS worker returned HTTP {exc.code}.") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"TTS worker is unavailable at {self.base_url}: {exc.reason}") from exc

        output_path.write_bytes(audio_bytes)
        return {
            "path": str(output_path),
            "speaker": resolved_speaker,
            "language": self.language,
        }


tts_service = TtsService()
