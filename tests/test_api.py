import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient
    from backend import api, inference


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class InferenceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_health_reports_status(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_load_endpoint_uses_service(self):
        mocked_result = {
            "loaded": True,
            "model_path": "/tmp/model",
            "max_seq_length": 2048,
            "load_in_4bit": False,
        }

        with patch.object(inference.service, "load_model", return_value=mocked_result) as mocked_load:
            response = self.client.post(
                "/load",
                json={"model_path": "/tmp/model", "max_seq_length": 2048, "load_in_4bit": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mocked_result)
        mocked_load.assert_called_once_with(
            model_path="/tmp/model",
            max_seq_length=2048,
            load_in_4bit=False,
        )

    def test_generate_endpoint_passes_history_through(self):
        mocked_result = {
            "response": "Hello.",
            "history": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello."},
            ],
            "model_loaded": True,
        }

        with patch.object(inference.service, "generate", return_value=mocked_result) as mocked_generate:
            response = self.client.post(
                "/generate",
                json={"message": "Hi", "history": [], "max_new_tokens": 128},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mocked_result)
        mocked_generate.assert_called_once_with(
            message="Hi",
            history=[],
            max_new_tokens=128,
            system_prompt=None,
        )

    def test_generate_stream_endpoint_returns_sse(self):
        streamed_chunks = iter([
            'event: delta\ndata: {"delta": "He", "response": "He"}\n\n',
            'event: done\ndata: {"response": "Hello", "history": []}\n\n',
        ])

        with patch.object(inference.service, "stream_generate", return_value=streamed_chunks) as mocked_stream:
            with self.client.stream(
                "POST",
                "/generate/stream",
                json={"message": "Hi", "history": [], "max_new_tokens": 128},
            ) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: delta", body)
        self.assertIn("event: done", body)
        mocked_stream.assert_called_once_with(
            message="Hi",
            history=[],
            max_new_tokens=128,
            system_prompt=None,
        )

    def test_tts_endpoint_returns_wav_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            wav_path.write_bytes(b"RIFFdemo")

            with patch.object(
                api.tts_service,
                "synthesize",
                return_value={"path": str(wav_path), "speaker": "Alison Dietlinde", "language": "en"},
            ) as mocked_tts:
                response = self.client.post("/tts", json={"text": "Hello.", "speaker": "Alison Dietlinde"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-amadeus-speaker"], "Alison Dietlinde")
        mocked_tts.assert_called_once_with(text="Hello.", speaker="Alison Dietlinde")


if __name__ == "__main__":
    unittest.main()
