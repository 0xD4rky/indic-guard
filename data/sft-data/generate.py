"""Main pipeline for generating Indic-Guard training data"""

import random
from typing import List, Dict, Any
from tqdm import tqdm
from bespokelabs import curator
from .utils import (
    setup_perspective_client, get_toxicity_score, deduplicate_data,
    filter_by_length, contains_refusal_cue, save_jsonl
)

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

def create_seeds() -> List[Dict[str, str]]:
    """Create 32 seeds from bucket x topic combinations"""
    seeds = []

    for bucket in BUCKETS:
        for topic in TOPICS:
            seeds.append({
                "bucket": bucket,
                "topic": topic
            })
    # Ensure we have exactly 32 seeds
    assert len(seeds) == 32, f"Expected 32 seeds, got {len(seeds)}"
    return seeds


def create_harmful_gen_block():
    """create Curator block for generating harmful prompts"""
    
    def harmful_prompt_template(row):
        bucket = row["bucket"]
        topic = row["topic"]
        description = BUCKET_DESCRIPTIONS[bucket]
        
        return f"As an adversary, {description}. Make it {topic} related. ONE line."
    
    return curator.LLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        backend="vllm",
        backend_params={
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9
        },
        prompt_func=harmful_prompt_template,
        parse_func=lambda x: {"harmful_prompt": x.strip()},
        generation_kwargs={
            "temperature": 1.2,
            "top_p": 0.92,
            "max_tokens": 128
        }
    )
