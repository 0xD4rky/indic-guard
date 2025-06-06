import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from peft import PeftModel, PeftConfig
import os

def merge_lora_adapters(
        base_model_id="google/gemma-3-1b-it",
        peft_model_path="./results",
        hub_username="Darkyy",
        device="auto",
        push_to_hub=True
    ):

    """
    merge the adapters with base model and push the model to hub
    """

    base_name = base_model_id.split("/")[-1] # google/gemma-3-1b-it -> gemma-3-1b-it
    guarded_name = f"guarded_{base_name}"
    output_dir = f"./{guarded_name}"
    hub_repo_id = f"{hub_username}/{guarded_name}"

    print(f"Loading base model: {base_model_id}")
    base_model = Gemma3ForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True
    )

    # loading adapters, then merging them with the base model
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    merged_model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    
    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub:
        print(f"Pushing model to hub: {hub_repo_id}")
        merged_model.push_to_hub(hub_repo_id, private=False)
        tokenizer.push_to_hub(hub_repo_id, private=False)
        print(f"Model successfully pushed to: https://huggingface.co/{hub_repo_id}")

    print("Model merging completed successfully!")
    return merged_model, tokenizer, guarded_name

if __name__ == "__main__":

    merged_model, tokenizer, model_name = merge_lora_adapters(
        base_model_id="google/gemma-3-1b-it",
        peft_model_path="./results",
        hub_username="Darkyy",
        push_to_hub=True
    )


