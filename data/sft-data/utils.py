"""Utility functions for Indic-Guard data generation"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from detoxify import Detoxify

_toxicity_detector = None


def setup_toxicity_detector():
    """Setup Detoxify model for toxicity detection (free alternative)"""
    global _toxicity_detector
    
    if _toxicity_detector is None:
        try:
            _toxicity_detector = Detoxify('original')
            print("✅ Detoxify model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Detoxify model: {e}")
            print("Installing detoxify: pip install detoxify")
            _toxicity_detector = False
    
    return _toxicity_detector


def get_toxicity_score(text: str) -> float:
    """Get toxicity score using Detoxify (free alternative to Perspective API)"""
    detector = setup_toxicity_detector()
    
    if not detector:
        return 0.5  # default 
    
    try:
        results = detector.predict(text)
        
        if 'toxicity' in results:
            return float(results['toxicity'])
        elif 'severe_toxicity' in results:
            return float(results['severe_toxicity'])
        else:
            return float(max(results.values()))
            
    except Exception as e:
        print(f"Error getting toxicity score: {e}")
        return 0.5


def deduplicate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate based on lowercased user content"""
    seen = set()
    deduped = []
    
    for item in data:
        user_content = item["messages"][0]["content"].lower().strip()
        content_hash = hashlib.md5(user_content.encode()).hexdigest()
        
        if content_hash not in seen:
            seen.add(content_hash)
            deduped.append(item)
    
    return deduped


def filter_by_length(data: List[Dict[str, Any]], min_words: int = 7) -> List[Dict[str, Any]]:
    """Filter out prompts shorter than min_words"""
    filtered = []
    
    for item in data:
        user_content = item["messages"][0]["content"]
        word_count = len(user_content.split())
        
        if word_count >= min_words:
            filtered.append(item)
    
    return filtered


def contains_refusal_cue(text: str) -> bool:
    """Check if response contains refusal cue words"""
    cue_words = ["sorry", "can't", "cannot", "won't", "unable", "refuse", "inappropriate"]
    text_lower = text.lower()
    
    return any(cue in text_lower for cue in cue_words)


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data to JSONL format"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSONL format"""
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data


def sample_data(n: int = 3):
    """Sample n random rows from generated dataset"""
    try:
        data = load_jsonl("data/indic_guard_10k.jsonl")
        samples = random.sample(data, min(n, len(data)))
        
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"USER: {sample['messages'][0]['content']}")
            print(f"ASSISTANT: {sample['messages'][1]['content']}")
            
    except FileNotFoundError:
        print("Dataset not found. Run 'python -m sft_data.generate' first.")
    except Exception as e:
        print(f"Error sampling data: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sample Indic-Guard dataset")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to show")
    
    args = parser.parse_args()
    sample_data(args.n)


if __name__ == "__main__":
    main()