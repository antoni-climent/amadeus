# 🎻 Amadeus

By using Domain Adaptation on Qwen 3.5-4B, we make the model speak in the style of Amadeus, the AI that appears in the Steins;Gate 0 series.

For that, we got a dataset from the VN, and use unsloth for the fine-tuning.

## Training scripts
The files in ```src/``` are training and experimentation scripts. ```DAPT.py``` runs domain adaptation training and ```telegram_bot.py``` starts a bot for Telegram interaction.

## App backend
The app-facing inference service lives in ```backend/inference.py```, and the FastAPI server lives in ```backend/api.py```.

## Data
Inside ```/data``` there is the data used for the training + an attempt to create synthetic data, where a model was prompted to generate the questions from the answers in the dataset. Here LLMs have shown to be really bad at predicting what comes previously to a sentence, so it was discarded.

## Model
The trained model is inside ```/qwen3.5-4b-kurisu-sg-corpus_v4```

## Next steps
- Create a desktop app to allow interaction with the model
- Add a React frontend and avatar layer on top of the backend API
 
