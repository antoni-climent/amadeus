import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.tts import TtsService


class DummyResponse:
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TtsServiceTests(unittest.TestCase):
    def test_synthesize_requires_text(self):
        service = TtsService(output_dir=Path("/tmp/output"))

        with self.assertRaisesRegex(ValueError, "Text is required"):
            service.synthesize("   ")

    def test_synthesize_writes_audio_from_worker_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "audio"
            service = TtsService(
                output_dir=output_dir,
                speaker="Ono_anna",
                language="English",
                base_url="http://127.0.0.1:8001",
            )

            with patch("backend.tts.urllib_request.urlopen", return_value=DummyResponse(b"WAVDATA")) as mocked_open:
                result = service.synthesize("Hello.")
                written_bytes = Path(result["path"]).read_bytes()

        self.assertEqual(result["speaker"], "Ono_anna")
        self.assertEqual(result["language"], "English")
        self.assertTrue(result["path"].endswith(".wav"))
        self.assertEqual(written_bytes, b"WAVDATA")
        request = mocked_open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8001/synthesize")
        self.assertEqual(request.get_method(), "POST")


if __name__ == "__main__":
    unittest.main()
