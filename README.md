# 🎻 Amadeus

By using Domain Adaptation on Qwen 3.5-4B, we make the model speak in the style of Amadeus, the AI that appears in the Steins;Gate 0 series.

For that, we got a dataset from the VN, and use unsloth for the fine-tuning.

## Training scripts
The files in ```src/``` are training and experimentation scripts. ```DAPT.py``` runs domain adaptation training and ```telegram_bot.py``` starts a bot for Telegram interaction.

## App backend
The app-facing inference service lives in ```backend/inference.py```, and the FastAPI server lives in ```backend/api.py```.

## TTS setup
The app uses 2 Python environments:
- `.amadeus_env`: main backend + Qwen 3.5 inference
- `.tts_env`: isolated Qwen3-TTS runtime

This split is necessary because the LLM stack and Qwen3-TTS require different `transformers` compatibility ranges.

The backend calls TTS through a persistent local worker running from `.tts_env`.
That worker keeps Qwen3-TTS loaded between requests, so synthesis is much faster after the first load.

Setup:

```bash
python3.11 -m venv .amadeus_env
source .amadeus_env/bin/activate
pip install -r requirements.txt

python3.12 -m venv .tts_env
source .tts_env/bin/activate
pip install -r requirements-tts.txt

cd frontend
npm install
cd ..
```

Run everything at once:

```bash
./run-dev.sh
```

This starts:
- FastAPI backend from `.amadeus_env`
- persistent Qwen3-TTS worker from `.tts_env`
- Vite frontend from `frontend/`

If you want to run only the backend + TTS worker:

```bash
source .tts_env/bin/activate
python -m backend.tts_worker

source .amadeus_env/bin/activate
export AMADEUS_TTS_URL="http://127.0.0.1:8001"
python backend/api.py
```

It also stores its runtime data inside the project:
- `.runtime/tts_home`
- `.runtime/mplconfig`
- `.runtime/amadeus_tts`

You can override these locations with:
- `AMADEUS_TTS_HOME`
- `AMADEUS_TTS_MPLCONFIGDIR`
- `AMADEUS_TTS_OUTPUT_DIR`

## Data
Inside ```/data``` there is the data used for the training + an attempt to create synthetic data, where a model was prompted to generate the questions from the answers in the dataset. Here LLMs have shown to be really bad at predicting what comes previously to a sentence, so it was discarded.

## Model
The trained model is inside ```/qwen3.5-4b-kurisu-sg-corpus_v4```

## Next steps
- Create a desktop app to allow interaction with the model
- Add a React frontend and avatar layer on top of the backend API
 
