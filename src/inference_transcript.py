from pathlib import Path
import re

from unsloth import FastVisionModel


LORA_FOLDER = Path(__file__).resolve().parent.parent / "models" / "qwen3.5-4b-dapt-kurisu_v15"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

SYSTEM_PROMPT = (
    "You are continuing a Steins;Gate multi-speaker transcript. "
    "Keep the existing order and tone. "
    "Every output line must use the format 'Speaker: text'. "
    "Use 'Narrator:' for narration and scene description. "
    "Preserve scene markers like 'Scene: SG00_01' when they are present in the prompt. "
    "Output only transcript lines."
)


def read_multiline_prompt():
    print("Paste transcript context. Finish with a line containing only END.")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    if not LORA_FOLDER.exists():
        raise FileNotFoundError(f"Trained model folder not found: {LORA_FOLDER}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(LORA_FOLDER),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    FastVisionModel.for_inference(model)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("--- Transcript Inference Started (type 'quit' to stop) ---")

    while True:
        prompt = read_multiline_prompt()
        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            print("Empty prompt. Paste transcript context or type quit.")
            continue

        user_prompt = (
            "Continue the transcript below.\n"
            "Keep speaker tags explicit and stay in transcript format.\n\n"
            f"{prompt}"
        )
        messages.append({"role": "user", "content": user_prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(text=[text], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        input_len = inputs.input_ids.shape[1]
        response_ids = outputs[0][input_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        print("\nAssistant:")
        print(response_clean)
        print()

        messages.append({"role": "assistant", "content": response_clean})
