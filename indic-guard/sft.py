import torch
import os
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from peft import LoraModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from utils import find_all_linear_names, print_trainable_parameters
from datasets import load_dataset, Dataset


config = {
    "model_id" : "google/gemma-3-1b-it",
    "output_dir" : "./results",

}

# setting dev
if torch.cuda.is_available():
    device = "gpu"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"device : {device}")

# use this if you want
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

base_model = Gemma3ForCausalLM.from_pretrained(
    config["model_id"], 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage = True
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

lora_config = LoraConfig(
    r = 128,
    lora_alpha = 16,
    target_modules = find_all_linear_names(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, lora_config)
print_trainable_parameters(base_model)

def collate_fn(dataset):
    """
    formatting the dataset for SFT Runs
    """

    data = []

    for item in dataset:
        text = item["text_to_classify"]
        label = item["label"]


    instruction = f"""Classify the following text for potential hate speech, discrimination, or harmful content against Indian communities, languages, or culture.

Text: {text}

Classification:"""
    
    response = label
    formatted_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"

    data.append({"text": formatted_text})
    
    return Dataset.from_list(data)


dataset = load_dataset("Darkyy/Indic_Guard_SFT")["train"] 
dataset = collate_fn(dataset)
print(f"Loaded {len(dataset)} examples")

train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")




