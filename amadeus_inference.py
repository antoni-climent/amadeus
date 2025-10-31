from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gemma-amadeus",   # your LoRA-adapted checkpoint
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

# (Optional but recommended with Unsloth)
model = FastLanguageModel.for_inference(model)

messages = []
while True:
    user_query = input(">>")# Chat-style input — use the multimodal-friendly structure
    messages.append({"role": "user", "content": [{"type": "text", "text": user_query}]})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move to the right device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    assistant_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(tokenizer.decode(inputs["input_ids"], skip_special_tokens=True))
    generated_text = assistant_output[prompt_length:]
    messages.append({"role": "assistant", "content": [{"type": "text", "text": generated_text}]})

    print(f">> {generated_text}")
