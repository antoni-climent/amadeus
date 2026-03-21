from unsloth import FastVisionModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from datasets import Dataset
import torch
import csv
import os
import sys

# ------------------------------------------------------------------------
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True   # Use 4bit quantization to reduce memory usage. Can be False.
# ------------------------------------------------------------------------

if __name__ == "__main__":
    
    model_id = "Qwen/Qwen3.5-4B"
    lora_folder = "../models/qwen3.5-4b-dapt-kurisu_v2"
    train_data_folder = "../data/VNScript/data.csv"

    lora_folder_name = lora_folder.split("/")[-1]

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not hasattr(model.config, "pad_token_id"):
        setattr(model.config, "pad_token_id", tokenizer.pad_token_id)
    elif model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Reset to training mode after baseline evaluation
    FastVisionModel.for_training(model)

    # Add LoRA adapters
    peft_args = {
        "finetune_vision_layers":     False, # False if not finetuning vision layers
        "finetune_language_layers":   True, # False if not finetuning language layers
        "finetune_attention_modules": True, # False if not finetuning attention layers
        "finetune_mlp_modules":       True, # False if not finetuning MLP layers
        "r": 16,
        #"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
    }
    model = FastVisionModel.get_peft_model(model, **peft_args)
    FastVisionModel.for_training(model)

    # Process  dataset
    formatted_samples = []
    with open(train_data_folder, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            messages = [
                {"role": "user", "content": row[1]},
                {"role": "assistant", "content": row[2]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            formatted_samples.append(text)

    dataset = Dataset.from_dict({"text": formatted_samples})

    # Training Arguments
    results_folder = os.path.join(lora_folder, "results")
    os.makedirs(results_folder, exist_ok=True)
    log_dir = os.path.join(results_folder, "logs")

    training_args = SFTConfig(
        output_dir = lora_folder,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.03,
        num_train_epochs = 2,
        learning_rate = 5e-6,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        packing = True, 
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        report_to = "wandb",
        logging_dir = log_dir,
        save_strategy = "steps",
        save_steps = 500,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = training_args,
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu_stats.name}. Max memory = {gpu_stats.total_memory / 1e9:.3f} GB.")

    print("Starting training...")
    trainer.train()

    model.save_pretrained(lora_folder) 
    tokenizer.save_pretrained(lora_folder)

    # Save training info
    with open(os.path.join(lora_folder, "training_args.txt"), 'w') as f:
        f.write(f"PEFT Args: {peft_args}\n\nTraining Args: {training_args.to_dict()}\n")    
    print("Training completed.")
