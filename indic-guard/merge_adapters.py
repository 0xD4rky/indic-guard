
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import argparse

def merge_lora_adapters(
    base_model_id="sarvamai/sarvam-m",
    peft_model_path="./results",
    hub_username="Darkyy",
    device="auto",
    push_to_hub=True  
):
    """
    merge the adapters to the main model
    """
    
    base_name = base_model_id.split("/")[-1]
    guarded_name = f"guarded_{base_name}"
    output_dir = f"./{guarded_name}"
    hub_repo_id = f"{hub_username}/{guarded_name}"

    print(f"Loading base model: {base_model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2"
    )
    print(f"Loading LoRA adapters from: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    merged_model = model.merge_and_unload()

    # for some models, output contains generation config and temp, to handle that, implement this logic
    if hasattr(merged_model, 'generation_config') and merged_model.generation_config is not None:
        gen_config = merged_model.generation_config
        if hasattr(merged_model, 'temperature') and hasattr(gen_config, 'do_sample'):
            if gen_config.temperature is not None and gen_config.temperature > 0:
                gen_config.do_sample = True
            elif gen_config.do_sample is False:
                gen_config.temperature = None

    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    except ValueError as e:
        if "GenerationConfig is invalid" in str(e):
            print("Fixing generation config issue")
            merged_model.generation_config = None
            merged_model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
        else:
            raise e
        
    if push_to_hub:
        print(f"Pushing model to hub: {hub_repo_id}")
        try:
            merged_model.push_to_hub(hub_repo_id, private=False, safe_serialization=True)
            tokenizer.push_to_hub(hub_repo_id, private=False)
            print(f"Model successfully pushed to: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            print(f"Error pushing to hub: {e}")

    print("Model merging completed successfully")
    return merged_model, tokenizer, guarded_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="sarvamai/sarvam-m")
    parser.add_argument("--hub_username", type=str, default="Darkyy")
    parser.add_argument("--push_to_hub", type=str, default="True")
    args = parser.parse_args()

    merged_model, tokenizer, model_name = merge_lora_adapters(
        base_model_id=args.base_model_id,
        peft_model_path="./results",
        hub_username=args.hub_username,
        push_to_hub=args.push_to_hub.lower() == "true"
    )
