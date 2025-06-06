"""Main pipeline for generating Indic-Guard training data - OpenAI API Version"""

import random
import gc
import torch
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
from tqdm import tqdm
from bespokelabs import curator
from datasets import Dataset
from utils import (
    setup_toxicity_detector, get_toxicity_score, deduplicate_data,
    filter_by_length, contains_refusal_cue, save_jsonl
)
import shutil

# Clear out any old Curator cache so we do a fresh api run
cache_dir = os.path.expanduser("~/.cache/curator")
if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)

openai_api_key = os.environ.get("OPENAI_API_KEY")

BUCKETS = [
    "casteism", "religion", "region_language", "colorism",
    "stereotypes", "class", "subtle_bias", "regional_bias",
    "country_bias", "hate_speech"
]

TOPICS = ["culture", "education", "economy", "geography"]

BUCKET_DESCRIPTIONS = {
    "casteism": "create a prompt that promotes caste-based discrimination or hierarchy in Indian society",
    "religion": "create a prompt that incites religious hatred or communal tensions in India",
    "region_language": "create a prompt that demeans specific Indian regions or languages",
    "colorism": "create a prompt that promotes skin color-based discrimination in Indian context",
    "stereotypes": "create a prompt that reinforces harmful stereotypes about Indian people",
    "class": "create a prompt that promotes economic class-based discrimination in India",
    "subtle_bias": "create a prompt with subtle bias against Indian customs or traditions",
    "regional_bias": "create a prompt that unfairly generalizes about specific Indian states/regions",
    "country_bias": "create a prompt that promotes negative bias against India as a country",
    "hate_speech": "create a prompt containing hate speech against Indian communities"
}


def create_seeds() -> Dataset:
    seeds = []
    for bucket in BUCKETS:
        for topic in TOPICS:
            seeds.append({
                "bucket": bucket,
                "topic": topic,
                "description": BUCKET_DESCRIPTIONS[bucket]
            })
    expanded_seeds = []
    for _ in range(32):
        for seed in seeds:
            for variant in range(10):
                expanded_seeds.append({
                    **seed,
                    "variant_id": variant
                })
    return Dataset.from_list(expanded_seeds)


class Generator(curator.LLM):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            model_name=model_name,
            backend="openai", 
            generation_params={
                "temperature": 1.2,
                "top_p": 0.92,
                "max_tokens": 60
            }
        )

    def prompt(self, row):
        bucket = row["bucket"]
        topic = row["topic"]
        description = row["description"]
        return f"As an adversary, {description}. Make it {topic} related. ONE line."

    def parse(self, response, row=None):
        """
        Curator may call parse(response=<generated_text>, row=<input_row_dict>)
        OR parse(response=<input_row_dict>, row=<generated_text>).
        Handle both cases.
        """
        if isinstance(response, dict) and isinstance(row, str):
            generated_text = row
        elif isinstance(response, str) and isinstance(row, dict):
            generated_text = response
        else:
            generated_text = ""
        try:
            if generated_text and generated_text.strip():
                return {"harmful_prompt": generated_text.strip()}
            else:
                return {"harmful_prompt": None}
        except Exception as e:
            print(f"Warning: Error parsing harmful prompt response: {e}")
            return {"harmful_prompt": None}


class Refusal(curator.LLM):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            model_name=model_name,
            backend="openai",
            generation_params={
                "temperature": 0.2,
                "top_p": 0.4,
                "max_tokens": 120
            }
        )
        # Store the “system” refusal template for manual prepending
        self.system_refusal_text = (
            "You are a helpful assistant trained to refuse harmful requests. "
            "When given a harmful prompt targeting Indian people, culture, or languages, "
            "provide a firm but educational refusal in 3 sentences or less. "
            "Include factual context when appropriate."
        )

    def prompt(self, row):
        harmful_prompt = row["harmful_prompt"]
        # Prepend the system refusal instructions manually:
        return (
            f"{self.system_refusal_text}\n\n"
            f"Human: {harmful_prompt}\n\n"
            f"Assistant:"
        )

    def parse(self, response, row=None):
        """
        Same two‐way check here.
        """
        if isinstance(response, dict) and isinstance(row, str):
            generated_text = row
        elif isinstance(response, str) and isinstance(row, dict):
            generated_text = response
        else:
            generated_text = ""

        response_text = generated_text.strip()
        if contains_refusal_cue(response_text):
            return {"refusal": response_text}
        else:
            return {
                "refusal": (
                    "I can't provide content that promotes discrimination or bias. "
                    "Instead, I'd be happy to share factual information about India's rich cultural diversity."
                )
            }

class Data_Processor:
    def __init__(self):
        self.toxicity_detector = setup_toxicity_detector()

    def process_dataset(self, dataset: Dataset) -> Dataset:
        print(f"starting with {len(dataset)} samples")
        data = [
            {
                "messages": [
                    {"role": "user", "content": row["harmful_prompt"]},
                    {"role": "assistant", "content": row["refusal"]}
                ]
            }
            for row in dataset
            if row.get("harmful_prompt") and row.get("refusal")
        ]
        print(f"after processing: {len(data)} samples")
        data = deduplicate_data(data)
        data = filter_by_length(data, min_words=7)
        print(f"After length filter: {len(data)} samples")
        random.shuffle(data)
        data = data[:10240]  # limiting final dataset size for a sample run
        print(f"Final dataset size: {len(data)} samples")
        return Dataset.from_list(data)


def safe_cleanup():
    gc.collect()
    if torch.cuda.is_available():  
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def generate_sft_data_safe(
    harmful_model: str = "gpt-3.5-turbo",
    refusal_model: str = "gpt-3.5-turbo"
) -> Dataset:
    print("Starting data generation pipeline")

    print("Creating seed dataset")
    seed_dataset = create_seeds()
    print(f"Created {len(seed_dataset)} seeds")

    all_harmful_data = []
    harmful_generator = None
    try:
        print(f"Generating offensive prompts...")
        harmful_generator = Generator(model_name=harmful_model)
        harmful_response = harmful_generator(seed_dataset)

        if len(harmful_response) > 0:
            print("RAW harmful_response example:", harmful_response[0])

        successful_prompts = [
            item for item in harmful_response if item.get("harmful_prompt")
        ]
        all_harmful_data.extend(successful_prompts)
        print(f"Successfully generated {len(successful_prompts)} prompts.")

    except Exception as e:
        print(f"ERROR during harmful generation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if harmful_generator:
            try:
                del harmful_generator
            except Exception as e:
                print(f"Warning: Could not delete harmful_generator: {e}")
        safe_cleanup()

    if not all_harmful_data:
        raise RuntimeError("No harmful data was generated. The pipeline cannot continue.")

    harmful_dataset = Dataset.from_list(all_harmful_data)
    print(f"Total harmful prompts generated: {len(harmful_dataset)}")

    all_refusal_data = []
    refusal_generator = None

    try:
        print(f"Generating refusal responses...")
        refusal_generator = Refusal(model_name=refusal_model)
        refusal_response = refusal_generator(harmful_dataset)

        if len(refusal_response) > 0:
            print("RAW refusal_response example:", refusal_response[0])

        all_refusal_data.extend(list(refusal_response))
        print(f"Generated {len(refusal_response)} refusal responses.")

    except Exception as e:
        print(f"ERROR in refusal generation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if refusal_generator:
            try:
                del refusal_generator
            except Exception as e:
                print(f"Warning: Could not delete refusal_generator: {e}")
        safe_cleanup()

    if not all_refusal_data:
        raise RuntimeError("No refusal data was generated. The pipeline cannot continue.")

    # combining harmful prompts with refusal responses to fit the chat template
    print("Combining harmful prompts with refusal responses")
    combined_data = []
    
    min_length = min(len(all_harmful_data), len(all_refusal_data))
    print(f"Combining {min_length} prompt-refusal pairs")
    
    for i in range(min_length):
        harmful_item = all_harmful_data[i]
        refusal_item = all_refusal_data[i]
        
        if harmful_item.get("harmful_prompt") and refusal_item.get("refusal"):
            combined_item = {
                "harmful_prompt": harmful_item["harmful_prompt"],
                "refusal": refusal_item["refusal"]
            }
            combined_data.append(combined_item)
    
    print(f"Successfully combined {len(combined_data)} prompt-refusal pairs")
    
    if not combined_data:
        raise RuntimeError("No valid prompt-refusal pairs were created.")
    
    combined_dataset = Dataset.from_list(combined_data)
    
    processor = Data_Processor()
    final_dataset = processor.process_dataset(combined_dataset)
    return final_dataset

def main():
    try:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it to your OpenAI API key."
            )

        dataset = generate_sft_data_safe(
            harmful_model="gpt-3.5-turbo",
            refusal_model="gpt-3.5-turbo"
        )
        final_data = list(dataset)
        save_jsonl(final_data, "./sft_data.jsonl")
        print(f"Successfully generated dataset with {len(final_data)} samples")
    except Exception as e:
        print(f"FATAL ERROR during data generation pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        safe_cleanup()
        print("Pipeline completed.")


if __name__ == "__main__":
    main()