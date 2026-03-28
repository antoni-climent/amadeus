import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.inference import (
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SEQ_LENGTH,
    service,
)
from backend.tts import DEFAULT_TTS_SPEAKER, tts_service


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=DEFAULT_MAX_NEW_TOKENS)
    system_prompt: str | None = None


class LoadRequest(BaseModel):
    model_path: str | None = None
    max_seq_length: int = Field(default=DEFAULT_MAX_SEQ_LENGTH, ge=1)
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker: str = Field(default=DEFAULT_TTS_SPEAKER, min_length=1)


app = FastAPI(title="Amadeus Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": service.is_loaded(),
        "model_path": str(service.model_path),
    }


@app.post("/load")
def load_model(request: LoadRequest) -> dict[str, object]:
    try:
        return service.load_model(
            model_path=request.model_path,
            max_seq_length=request.max_seq_length,
            load_in_4bit=request.load_in_4bit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, object]:
    try:
        return service.generate(
            message=request.message,
            history=request.history,
            max_new_tokens=request.max_new_tokens,
            system_prompt=request.system_prompt,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest) -> StreamingResponse:
    try:
        stream = service.stream_generate(
            message=request.message,
            history=request.history,
            max_new_tokens=request.max_new_tokens,
            system_prompt=request.system_prompt,
        )
        return StreamingResponse(stream, media_type="text/event-stream")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


@app.post("/tts")
def tts(request: TtsRequest) -> FileResponse:
    try:
        result = tts_service.synthesize(text=request.text, speaker=request.speaker)
        return FileResponse(
            path=result["path"],
            media_type="audio/wav",
            filename=Path(result["path"]).name,
            headers={"X-Amadeus-Speaker": result["speaker"]},
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("AMADEUS_HOST", "127.0.0.1")
    port = int(os.getenv("AMADEUS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)
