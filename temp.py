from datasets import load_dataset
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

print(dataset[:3])
# Convert to Unsloth's conversation format
# dataset = dataset.map(
#     lambda x: {
#         "conversations": [
#             {"role": "user", "value": x["user"]},
#             {"role": "assistant", "value": x["assistant"]}
#         ]
#     }
# )
#
# # Standardize for Unsloth fine-tuning
# dataset = standardize_data_formats(dataset)
#
# print(dataset[0])


