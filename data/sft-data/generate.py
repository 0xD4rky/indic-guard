"""Main pipeline for generating Indic-Guard training data"""

import random
from typing import List, Dict, Any
from tqdm import tqdm
from bespokelabs import curator
from datasets import Dataset
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
                "topic": topic,
                "description": BUCKET_DESCRIPTIONS[bucket]
            })
    expanded_seeds = []
    for _ in range(32):  # 32 iterations
        for seed in seeds:
            for variant in range(10):  # 10 variants per seed
                expanded_seeds.append({
                    **seed,
                    "variant_id": variant
                })
    
    return Dataset.from_list(expanded_seeds)




