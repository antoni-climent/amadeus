# 🎻 Amadeus

By using Domain Adaptation on gemma3-4B, we make the model speak in the style of Amadeus, the AI that appears in the Steins;Gate 0 series.

For that, we got a dataset from the https://github.com/Ibnelaiq/KurisuQA repository, and used trl for the fine-tuning.

## Training scripts
The ```DAPT.py``` makes the training, ```inference.py``` implements the local inference, and ```telegram_bot.py``` starts a bot to allow interaction via the telegram app.

```unsloth_training.py``` and ```sft_trl_lora_qlora.ipynb``` are tries to train the model by using pairs of question-answer, but due to the data quality of the pairs it was discarded, and only the answers were used.

## Data
Inside ```/data``` there is the data used for the training + an attempt to create synthetic data, where a model was prompted to generate the questions from the answers in the dataset. Here LLMs have shown to be really bad at predicting what comes previously to a sentence, so it was discarded.

## Model
The trained model is inside ```/gemma3-4b-dapt-kurisu```
