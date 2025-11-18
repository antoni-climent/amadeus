# Domain-Adaptive Pre-Training (DAPT) on Kurisu VN dialogues

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import csv

model_id = "google/gemma-3-4b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
    dtype='auto',                          # Change to bfloat16 if GPU has support
    device_map='cuda:0',
    # use_cache=True,                               # Whether to cache attention outputs to speed up inference
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    )
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|kurisu|>"]})
model.resize_token_embeddings(len(tokenizer))

# Build DAPT dataset (from VNresponses only)
text_samples = []
with open('./data/VNKurisuDialogues.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        text_samples.append(f"<|kurisu|> {row[1]}")  # Append user message with special token

from datasets import Dataset
dataset = Dataset.from_dict({"text": text_samples})

# LoRA config
from peft import LoraConfig
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",] # ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] 
)

# SFT (DAPT) config
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    # Training schedule / optimization
    # assistant_only_loss=True,        # Compute loss only on assistant's tokens
    per_device_train_batch_size = 1,      # Batch size per GPU
    gradient_accumulation_steps = 2,      # Gradients are accumulated over multiple steps → effective batch size = 2 * 8 = 16
    warmup_ratio = 0.03,
    num_train_epochs = 1,               # Number of full dataset passes. For shorter training, use `max_steps` instead (this case)
    #max_steps = 30,
    learning_rate = 1e-5,                 # Learning rate for the optimizer
    optim = "paged_adamw_8bit",           # Optimizer

    # Logging / reporting
    logging_steps=5,                      # Log training metrics every N steps
    report_to="trackio",                  # Experiment tracking tool
    # trackio_space_id=output_dir,          # HF Space where the experiment tracking will be saved
    output_dir="qwen2.5-7b-dapt-kurisu",               # Where to save model checkpoints and logs
    dataset_text_field="text",
    max_length=2048,                      # Maximum input sequence length
    use_liger_kernel=True,                # Enable Liger kernel optimizations for faster training
    activation_offloading=True,           # Offload activations to CPU to reduce GPU memory usage
    gradient_checkpointing=False,          # Save memory by re-computing activations during backpropagation

    # Hub integration
    push_to_hub=False,                     # Automatically push the trained model to the Hugging Face Hub
                                          # The model will be saved under your Hub account in the repository named `output_dir`

    gradient_checkpointing_kwargs={"use_reentrant": False}, # To prevent warning message
)

trainer = SFTTrainer(
    model=model, args=training_args, processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config
)
trainer.train()
trainer.save_model("qwen2.5-7b-dapt-kurisu")
