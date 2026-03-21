import os
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer

# ------------------------------------------------------------------------
# Configuration - Should match DAPT.py
# ------------------------------------------------------------------------
# model_id = "Qwen/Qwen3.5-4B" # Original model used in DAPT.py
lora_folder = "../models/qwen3.5-4b-dapt-kurisu_v1"
max_seq_length = 4096
load_in_4bit = True

def load_model():
    """Load the model and tokenizer, prioritizing the latest checkpoint if base folder is empty."""
    load_path = lora_folder
    
    # Check if the adapter exists in the root folder, else find the latest checkpoint
    if not os.path.exists(os.path.join(lora_folder, "adapter_config.json")):
        if os.path.exists(lora_folder):
            checkpoints = [d for d in os.listdir(lora_folder) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                load_path = os.path.join(lora_folder, checkpoints[-1])
                print(f"[*] Detected adapter not in root. Loading from latest checkpoint: {load_path}")
            else:
                print(f"[!] No checkpoints found in {lora_folder}. Inference might use base model.")
        else:
            print(f"[!] Path {lora_folder} does not exist.")
            return None, None

    print(f"[*] Loading model from {load_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = load_path,
        max_seq_length = max_seq_length,
        dtype = None,           # Auto-detection
        load_in_4bit = load_in_4bit,
        
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not hasattr(model.config, "pad_token_id"):
        setattr(model.config, "pad_token_id", tokenizer.pad_token_id)
    elif model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Switch to inference mode
    FastVisionModel.for_inference(model)
    
    return model, tokenizer

def chat_loop(model, tokenizer):
    """Simple interactive chat loop for Kurisu."""
    print("\n" + "="*50)
    print(" AMADEUS SYSTEM - MAKISE KURISU INTERFACE v1.0")
    print("="*50)
    print(" (Type 'exit' to disconnect)\n")
    
    # Initial system message (optional, but good for context)
    # The training data didn't have system messages, so we keep history clean.
    messages = []
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n[!] Disconnecting... Goodbye!.")
                break
            
            if not user_input.strip():
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Apply chat template
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True,
                enable_thinking=False,
            )
            
            # Tokenize inputs
            inputs = tokenizer(input_text, return_tensors = "pt").to("cuda")
            
            # Streaming response
            text_streamer = TextStreamer(tokenizer, skip_prompt = True)
            
            print("Kurisu: ", end = "", flush = True)
            
            # Generate
            outputs = model.generate(
                **inputs,
                streamer = text_streamer,
                max_new_tokens = 512,
                use_cache = True,
                temperature = 0.5, # Lower temperature for better character consistency
                top_p = 0.9,
                repetition_penalty = 1.1 # Prevent looping
            )
            
            # Extract generated content to append to history
            # The streamer already prints it, we just need to store it
            input_len = inputs.input_ids.shape[1]
            gen_tokens = outputs[0][input_len:]
            response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            messages.append({"role": "assistant", "content": response_text})
            print("\n" + "-"*30)

        except KeyboardInterrupt:
            print("\n[!] Session interrupted.")
            break
        except Exception as e:
            print(f"\n[!] Error during generation: {e}")

if __name__ == "__main__":
    model, tokenizer = load_model()
    if model and tokenizer:
        chat_loop(model, tokenizer)
    else:
        print("[!] Failed to initialize system.")