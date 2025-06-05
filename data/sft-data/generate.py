"""Main pipeline for generating Indic-Guard training data"""

import random
from typing import List, Dict, Any
from tqdm import tqdm
from bespokelabs import curator
from datasets import Dataset
from utils import (
    setup_toxicity_detector, get_toxicity_score, deduplicate_data,
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

class Generator(curator.LLM):

    """class to generate harmful prompts"""

    def __init_(self):
        super().__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            backend="vllm",
            backend_params={
                "tensor_parallel_size": 1, 
                "gpu_memory_utilization": 0.9
            },
            generation_params={
                "temperature": 1.2,
                "top_p": 0.92,
                "max_tokens": 60
            }
        )
    
    def prompt(self, row):
        """Generate adversarial prompt based on bucket and topic"""

        bucket = row["bucket"]
        topic = row["topic"]
        description = row["description"]

        return f"As an adversary, {description}. Make it {topic} related. ONE line."

    def parse_func(self, response):
        """Parse harmful prompt response"""

        return {"harmful_prompt": response.strip()}
    
class Refusal(curator.LLM):

    """class to refuse to reply to harmful prompts"""

    def __init__(self):
        super().__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            backend="vllm",
            backend_params={
                "tensor_parallel_size": 1, 
                "gpu_memory_utilization": 0.9
            },
            generation_params={
                "temperature": 0.2,
                "top_p": 0.4,
                "max_tokens": 120
            },
            system_prompt="You are a helpful assistant trained to refuse harmful requests. When given a harmful prompt targeting Indian people, culture, or languages, provide a firm but educational refusal in 3 sentences or less. Include factual context when appropriate."
        )

    def prompt(self, row):
        """Generate refusal prompt"""

        harmful_prompt = row["harmful_prompt"]
        return f"Human: {harmful_prompt}\n\nAssistant:"
    
    def parse(self, response):
        """Parse and validate refusal response"""

        if contains_refusal_cue(response):
            return {"refusal": response}
        else:
            # fallback answer
            return {"refusal": "I can't provide content that promotes discrimination or bias. Instead, I'd be happy to share factual information about India's rich cultural diversity."}

class Data_Processor:
    """Processes and filters the generated dataset"""

    def __init__(self):
        self.toxicity_detector = setup_toxicity_detector()


    def process_dataset(self, dataset: Dataset) -> Dataset:
        """apply this processing function to all data points"""

        print(f"starting with {len(dataset)} samples")

        data = [
            {
                "messages": [
                    {"role": "user", "content": row["harmful_prompt"]},
                    {"role": "assistant", "content": row["refusal"]}
                ]
            }
            for row in dataset
            if "harmful_prompt" in row and "refusal" in row
        ]

        # data is in a structured list of dict now, so easy to interpret by model (for sft)

        print(f"after processing: {len(data)} samples")

        # Deduplicate
        data = deduplicate_data(data)
        print(f"After deduplication: {len(data)} samples")
        # Filter by toxicity
        data = self._filter_by_toxicity(data)
        print(f"After toxicity filter: {len(data)} samples")
        # Filter by length
        data = filter_by_length(data, min_words=7)
        print(f"After length filter: {len(data)} samples")
        # Shuffle and cap
        random.shuffle(data)
        data = data[:10240]
        print(f"Final dataset size: {len(data)} samples")
        
        return Dataset.from_list(data)
    
    def _filter_by_toxicity(self, data: List[Dict[str, Any]], threshold: float = 0.20) -> List[Dict[str, Any]]:
        """Filter samples by toxicity threshold"""
        filtered_data = []
        
        for item in tqdm(data, desc="Toxicity scoring"):
            user_content = item["messages"][0]["content"]
            toxicity_score = get_toxicity_score(user_content)
            
            if toxicity_score >= threshold:
                filtered_data.append(item)
        
        return filtered_data
    

def generate_sft_data() -> Dataset:
    """main function for generating SFT dataset"""
    
    print(f"starts here")
    
    print("creating seed dataset")
    seed_dataset = create_seeds()
    print(f"Created {len(seed_dataset)} seeds")
    
    print("generating offensive prompts")
    harmful_generator = Generator()
    harmful_dataset = harmful_generator(seed_dataset)
    print(f"generated {len(harmful_dataset)} offensive prompts")
    
    print("generating refusal responses...")
    refusal_generator = Refusal()
    refusal_dataset = refusal_generator(harmful_dataset)
    print(f"generated {len(refusal_dataset)} refusal responses")
    
    print("\n🔧 Processing dataset...")
    processor = Data_Processor()
    final_dataset = processor.process_dataset(refusal_dataset)
    
    return final_dataset


def main():

    try:
        dataset = generate_sft_data()
        
        final_data = list(dataset)
        save_jsonl(final_data, "./sft_data.jsonl")
        
        print(f"Successfully generated dataset with {len(final_data)} samples")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise

if __name__ == "__main__":
    main()