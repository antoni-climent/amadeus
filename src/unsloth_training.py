from unsloth import FastModel 
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset 
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# Load your CSV file
dataset = load_dataset("csv", data_files="data.csv")["train"]

def format_for_gemma(example):
    user_msg = example["user"].strip()
    assistant_msg = example["assistant"].strip()

    example["text"] = (
        f"<start_of_turn>user\n{user_msg}\n<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant_msg}\n<end_of_turn>"
    )
    return example

dataset = dataset.map(format_for_gemma)
dataset = dataset.remove_columns(["user", "assistant"])


model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# Add LORA modules 
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "gemma-3",
# )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        save_steps=300,
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train()

model.save_pretrained("gemma-amadeus")  # Local saving
tokenizer.save_pretrained("gemma-amadeus")


