from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

model_id, output_dir = "google/gemma-3-4b-it", "gemma3-4b-dapt-kurisu"

# base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="cuda")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
    dtype='auto',                          # Change to bfloat16 if GPU has support
    device_map='cuda',
    # use_cache=True,                               # Whether to cache attention outputs to speed up inference
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    )

)
fine_tuned_model = PeftModel.from_pretrained(base_model, output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|kurisu|>"]})

# messages = [
#     {
#         'content': "<|kurisu|> You are Makise Kurisu, a genius neuroscientist who graduated from Viktor Chondria University at 17. \
#     You harbor a soft, flustered side that surfaces when teased or emotionally exposed. \
#     You enjoy intellectual discussions, debates, and dismantling flawed logic with cutting precision. \
#     You care deeply for your friends. Your tone is warm and supportive.",
#         'role': 'system',
#     },
# ]
messages = [
    {"role": "system", "content": "<|kurisu|> You are Makise Kurisu, a genious neuroscientist that loves intellectual debates \
        and care deeply about friends. Your tone is warm, and supportive."},
]
# messages = [
#     {"role": "system", "content": "<|kurisu|> You respond in Makise Kurisu’s voice."},
# ]
    
while True:
    user_input = input("User: ")
    
    messages.append({
        'content': user_input,
        'role': 'user'
    })
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(fine_tuned_model.device)

    eos = tokenizer.eos_token_id
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    sot = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    terminators = [i for i in [eos, eot] if i is not None]

    gen_kwargs = dict(
        max_new_tokens=1024,
        # min_new_tokens=128,          # force it to keep going
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.05,     # gentle push against loops
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    generated_ids = fine_tuned_model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True, temperature=0.2, top_p=0.9,
        repetition_penalty=1.05, no_repeat_ngram_size=4,
        eos_token_id=terminators,  # stop on EOS or EOT
        pad_token_id=tokenizer.pad_token_id or eos,
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

    # Decode and extract model response
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    messages.append({
        'content': generated_text,
        'role': 'assistant'
    })

    print("Assistant: ", generated_text)
    print("----------------------------------------")