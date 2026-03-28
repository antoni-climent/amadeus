import io
import os

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


DEFAULT_TTS_MODEL_NAME = os.getenv("AMADEUS_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_TTS_SPEAKER = os.getenv("AMADEUS_TTS_SPEAKER", "Ono_anna")
DEFAULT_TTS_LANGUAGE = os.getenv("AMADEUS_TTS_LANGUAGE", "English")
DEFAULT_TTS_DEVICE = os.getenv("AMADEUS_TTS_DEVICE", "auto")
DEFAULT_TTS_INSTRUCT = os.getenv("AMADEUS_TTS_INSTRUCT", "")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker: str = Field(default=DEFAULT_TTS_SPEAKER, min_length=1)
    language: str = Field(default=DEFAULT_TTS_LANGUAGE, min_length=1)
    instruct: str | None = None


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


class PersistentTtsService:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def get_model(self):
        if self._model is None:
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                **resolve_model_kwargs(self.device),
            )
        return self._model

    def synthesize(self, text: str, speaker: str, language: str, instruct: str | None) -> bytes:
        model = self.get_model()
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct or None,
        )

        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sample_rate, format="WAV")
        return buffer.getvalue()


service = PersistentTtsService(DEFAULT_TTS_MODEL_NAME, DEFAULT_TTS_DEVICE)
app = FastAPI(title="Amadeus TTS Worker")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_name": service.model_name,
        "loaded": service._model is not None,
    }


@app.post("/synthesize")
def synthesize(request: SynthesizeRequest) -> Response:
    try:
        audio = service.synthesize(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            instruct=request.instruct if request.instruct is not None else DEFAULT_TTS_INSTRUCT,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-Amadeus-Speaker": request.speaker,
            "X-Amadeus-Language": request.language,
        },
    )


if __name__ == "__main__":
    import uvicorn

    service.get_model()
    host = os.getenv("AMADEUS_TTS_HOST", "127.0.0.1")
    port = int(os.getenv("AMADEUS_TTS_PORT", "8001"))
    uvicorn.run(app, host=host, port=port, reload=False)
