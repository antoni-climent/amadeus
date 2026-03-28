import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.tts import TtsService


class TtsServiceTests(unittest.TestCase):
    def test_synthesize_requires_text(self):
        service = TtsService(python_executable=Path("/tmp/python"), output_dir=Path("/tmp/output"))

        with self.assertRaisesRegex(ValueError, "Text is required"):
            service.synthesize("   ")

    def test_synthesize_requires_python_executable(self):
        service = TtsService(python_executable=Path("/tmp/does-not-exist"), output_dir=Path("/tmp/output"))

        with self.assertRaisesRegex(FileNotFoundError, "TTS Python executable does not exist"):
            service.synthesize("Hello.")

    def test_synthesize_runs_backend_tts_runner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            python_executable = Path(temp_dir) / "python"
            python_executable.write_text("", encoding="utf-8")
            output_dir = Path(temp_dir) / "audio"

            service = TtsService(
                python_executable=python_executable,
                output_dir=output_dir,
                tts_home=Path(temp_dir) / "tts_home",
                mplconfigdir=Path(temp_dir) / "mplconfig",
                speaker="Alison Dietlinde",
            )

            with patch("backend.tts.subprocess.run") as mocked_run:
                mocked_run.return_value.returncode = 0
                mocked_run.return_value.stdout = ""
                mocked_run.return_value.stderr = ""

                result = service.synthesize("Hello.")

        self.assertEqual(result["speaker"], "Alison Dietlinde")
        self.assertEqual(result["language"], "en")
        self.assertTrue(result["path"].endswith(".wav"))
        self.assertEqual(mocked_run.call_args.kwargs["env"]["COQUI_TOS_AGREED"], "1")
        self.assertIn("--speaker", mocked_run.call_args.args[0])
        self.assertIn("--device", mocked_run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
