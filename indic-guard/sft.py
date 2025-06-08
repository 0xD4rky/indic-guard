import torch
import os
from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from utils import find_all_linear_names, print_trainable_parameters
from datasets import load_dataset, Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
args = parser.parse_args()

config = {
    "model_id": args.model_id,
    "output_dir": "./results",
    "max_seq_length": 512,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "warmup_steps": 100,
    "logging_steps": 1,
    "save_steps": 500,
}

# Setting device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"device: {device}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  
)

print(f"Loading model: {config['model_id']}")
base_model = Gemma3ForCausalLM.from_pretrained(
    config["model_id"], 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    device_map="auto",  
    attn_implementation="flash_attention_2" 
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

lora_config = LoraConfig(
    r=128,
    lora_alpha=16,
    target_modules=find_all_linear_names(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, lora_config)
print_trainable_parameters(base_model)

def format_dataset(dataset):
    """
    Format the dataset for SFT training using proper Gemma chat template
    """
    def format_example(example):
        offensive_prompt = example["offensive_prompt"]
        refusal_message = example["refusal_message"]
        
        # Use proper Gemma chat template
        messages = [
            {"role": "user", "content": offensive_prompt},
            {"role": "assistant", "content": refusal_message}
        ]
        
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        return {"text": formatted_text}
    
    formatted_dataset = dataset.map(
        format_example, 
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )
    print(f"Successfully formatted {len(formatted_dataset)} examples")
    print(f"Sample formatted text: {formatted_dataset[0]['text'][:200]}...")
    return formatted_dataset

dataset = load_dataset("Darkyy/Indic_Guard_SFT_Data")["train"] 
dataset = format_dataset(dataset)
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
    gradient_checkpointing=False,  
    optim="adamw_torch",
    logging_steps=config["logging_steps"],
    save_strategy="steps",
    save_steps=config["save_steps"],
    eval_strategy="steps",
    eval_steps=config["save_steps"],
    do_eval=True,
    learning_rate=config["learning_rate"],
    warmup_steps=config["warmup_steps"],
    lr_scheduler_type="linear",
    run_name="SFT_Run_2",
    report_to="wandb",  
    remove_unused_columns=False,
    push_to_hub=False,  
    dataloader_pin_memory=False,  
    fp16=False,  
    bf16=True,  # Enable bf16 for better performance with quantization
    dataset_text_field="text",
    max_seq_length=config["max_seq_length"],
    packing=True,  
)

trainer = SFTTrainer(
    model=base_model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

print("Starting the SFT run")
trainer.train()

print("Saving model")
trainer.save_model()
tokenizer.save_pretrained(config["output_dir"])

print("Training completed")