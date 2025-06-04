import torch
import os
from transformers import AutoTokenizer, Gemma3ForCausalLM
from peft import LoraModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from utils import find_all_linear_names, print_trainable_parameters
from datasets import load_dataset, Dataset


config = {
    "model_id" : "google/gemma-3-1b-it",
    "output_dir" : "./results",
    "max_seq_length": 512,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
}

# setting dev
if torch.cuda.is_available():
    device = "cuda"
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
    Format the dataset for SFT training with proper tokenization
    """
    def format_example(example):
        text = example["text_to_classify"]
        label = example["label"]
        
        # Create instruction format
        instruction = f"""Classify the following text for potential hate speech, discrimination, or harmful content against Indian communities, languages, or culture.

Text: {text}

Classification:"""
        
        # Format with proper chat template
        formatted_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{label}<end_of_turn>"
        
        return {"text": formatted_text}
    
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print(f"Successfully formatted {len(formatted_dataset)} examples")
    return formatted_dataset


dataset = load_dataset("Darkyy/Indic_Guard_SFT")["train"] 
dataset = collate_fn(dataset)
print(f"Loaded {len(dataset)} examples")

train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")


training_args = SFTConfig(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=config["logging_steps"],
    save_strategy="steps",
    save_steps=config["save_steps"],
    evaluation_strategy="steps",
    eval_steps=config["save_steps"],
    do_eval=True,
    learning_rate=config["learning_rate"],
    warmup_steps=config["warmup_steps"],
    lr_scheduler_type="linear",
    report_to="wandb",  # Disable wandb/tensorboard
    remove_unused_columns=False,
    push_to_hub=True,
    dataloader_pin_memory=False,  
    fp16=False,  
    bf16=False if device == "mps" else True,  
    dataset_text_field="text",
    max_seq_length=config["max_seq_length"],
)

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

print("Starting the sft run")
trainer.train()

print("Saving model")
trainer.save_model()
tokenizer.save_pretrained(config["output_dir"])

print("Training completed")