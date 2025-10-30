from datasets import load_dataset
from unsloth import FastModel
import torch
from transformers import TextStreamer

# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

messages = [{
    "role" : "user",
    "content": [
        { "type": "text",  "text" : "Hello, how are you doing?" }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(messages, max_new_tokens = 256)





# from VNresponses import data
# import csv

# with open('data.csv', mode='w') as csvfile:
#     csv_writer = csv.writer(csvfile, delimiter=',')
#     for question in data.keys():
#         responses = ''.join([sentence + " " for sentence in data[question]]) 
#         csv_writer.writerow([question, responses])
