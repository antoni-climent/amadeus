import os
import torch
from unsloth import FastVisionModel

lora_folder = "../models/qwen3.5-4b-kurisu-sg-corpus_v4/checkpoint-300"
# "../models/qwen3.5-4b-dapt-kurisu_v12/checkpoint-80" is pretty good with new prompt
max_seq_length = 8192
load_in_4bit = True


if __name__ == "__main__":
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = lora_folder,
        max_seq_length = max_seq_length,
        dtype = None,           # Auto-detection
        load_in_4bit = load_in_4bit,
    )

    # Switch to inference mode
    FastVisionModel.for_inference(model)

    messages = []
    print("--- Conversation Started (type 'exit' or 'quit' to stop) ---")

    # Get system prompt from file
    with open("/home/toni/Desktop/system_prompt.txt", 'r') as file:
        system_prompt = file.read()
    # messages.append({"role": "system", "content": system_prompt})

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Try to suppress it in the prompt
        )

        inputs = tokenizer(text=[text], return_tensors="pt").to("cuda")

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        input_len = inputs.input_ids.shape[1]
        response_ids = outputs[0][input_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Remove thinking tokens/blocks if they still appear
        import re
        response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        print(f"Assistant: {response_clean}")

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response_clean})