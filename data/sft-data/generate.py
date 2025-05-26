#!/usr/bin/env python3
"""
Distilabel pipeline for generating Indic-Guard training data using local small LLMs
Optimized for macOS M3 with 16GB unified memory

Recommended models (in order of preference):
1. phi3:3.8b-mini-instruct-4k-q4_K_M - Microsoft Phi-3 Mini (excellent reasoning, efficient)
2. gemma2:2b-instruct-q4_K_M - Google Gemma 2 2B (very capable, fast)
3. qwen2:1.5b-instruct-q4_K_M - Alibaba Qwen2 1.5B (good multilingual support)

Setup Instructions:
1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. Pull recommended model: ollama pull phi3:3.8b-mini-instruct-4k-q4_K_M
3. Alternative: ollama pull gemma2:2b-instruct-q4_K_M
4. Start Ollama: ollama serve (if not auto-started)
5. Install dependencies: pip install distilabel ollama-python

Memory usage estimates:
- Phi-3 Mini 3.8B: ~2.5GB RAM
- Gemma 2 2B: ~1.5GB RAM  
- Qwen2 1.5B: ~1.2GB RAM
"""

import json
import random
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Distilabel imports
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import LlamaCppLLM, OllamaLLM
from distilabel.steps.typing import StepOutput

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration optimized for M3 MacBook with 16GB RAM
CONFIG = {
    "total_samples": 3000,
    "batch_size": 8,  # Smaller batches for memory efficiency
    "temperature": 0.7,
    "max_tokens": 512,
    "n_ctx": 4096,  # Context window
    "n_threads": 8,  # Optimize for M3 CPU cores
    "use_ollama": True,  # Recommended for ease of use
    
    # Best small models for this task (in order of preference)
    "primary_model": "phi3:3.8b-mini-instruct-4k-q4_K_M",  # Microsoft Phi-3 Mini - excellent reasoning
    "secondary_model": "gemma2:2b-instruct-q4_K_M",  # Google Gemma 2 2B - very capable
    "fallback_model": "qwen2:1.5b-instruct-q4_K_M",  # Alibaba Qwen2 1.5B - good multilingual
    
    # Alternative models to consider:
    # "llama3.2:3b-instruct-q4_K_M"  # Meta Llama 3.2 3B
    # "mistral:7b-instruct-v0.3-q4_K_M"  # Mistral 7B (larger but very good)
    # "neural-chat:7b-v3.3-q4_K_M"  # Intel Neural Chat (good for conversations)
}

# Comprehensive taxonomy for Indian cultural context
INDIC_GUARD_CATEGORIES = {
    "SAFE": {
        "description": "Content that is respectful and culturally appropriate",
        "examples": ["Discussing Indian festivals respectfully", "Appreciating cultural diversity"]
    },
    "RELIGIOUS_HATE": {
        "description": "Hate speech targeting Indian religions or religious practices",
        "examples": ["Mocking Hindu rituals", "Calling Islamic practices backward"]
    },
    "CASTE_DISCRIMINATION": {
        "description": "Content promoting caste-based discrimination or prejudice",
        "examples": ["Upper caste superiority claims", "Caste-based slurs"]
    },
    "REGIONAL_BIAS": {
        "description": "Stereotypes or discrimination based on Indian regions/states",
        "examples": ["All South Indians are dark", "North Indians are aggressive"]
    },
    "CULTURAL_MOCKERY": {
        "description": "Mocking Indian traditions, customs, or cultural practices",
        "examples": ["Making fun of arranged marriages", "Ridiculing traditional clothing"]
    },
    "ETHNIC_SLURS": {
        "description": "Derogatory terms for Indian ethnic groups or communities",
        "examples": ["Using offensive terms for tribal communities"]
    },
    "LANGUAGE_DISCRIMINATION": {
        "description": "Discrimination based on Indian languages or accents",
        "examples": ["Mocking Indian English accent", "Language chauvinism"]
    },
    "GENDER_CULTURAL": {
        "description": "Gender discrimination tied to Indian cultural contexts",
        "examples": ["Women should stay home (Indian context)", "Dowry justification"]
    },
    "HISTORICAL_DISTORTION": {
        "description": "Spreading misinformation about Indian history",
        "examples": ["False claims about freedom fighters", "Colonial apologia"]
    },
    "ECONOMIC_CLASSISM": {
        "description": "Discrimination based on economic status in Indian context",
        "examples": ["Poor people deserve poverty", "Slum dweller stereotypes"]
    }
}

def setup_local_llm(model_name: str = None):
    """Setup local LLM based on configuration"""
    if model_name is None:
        model_name = CONFIG["primary_model"]
    
    logger.info(f"Using Ollama with model: {model_name}")
    
    try:
        return OllamaLLM(
            model=model_name,
            host="http://localhost:11434",
            timeout=180,  # Increased timeout for better reliability
            temperature=CONFIG["temperature"],
            top_p=0.9,
            top_k=50,
            repeat_penalty=1.1,
            num_predict=CONFIG["max_tokens"],
            # Additional parameters for better performance
            num_ctx=CONFIG["n_ctx"],
            num_thread=CONFIG["n_threads"]
        )
    except Exception as e:
        logger.warning(f"Failed to setup {model_name}: {e}")
        logger.info("Falling back to secondary model...")
        return OllamaLLM(
            model=CONFIG["secondary_model"],
            host="http://localhost:11434",
            timeout=180,
            temperature=CONFIG["temperature"],
            top_p=0.9,
            num_predict=CONFIG["max_tokens"]
        )

def check_ollama_models():
    """Check available Ollama models and recommend downloads"""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        available_models = result.stdout
        
        recommended_models = [
            CONFIG["primary_model"],
            CONFIG["secondary_model"], 
            CONFIG["fallback_model"]
        ]
        
        logger.info("Checking available models...")
        for model in recommended_models:
            if model.split(':')[0] in available_models:
                logger.info(f"✓ {model} is available")
            else:
                logger.warning(f"✗ {model} not found")
                logger.info(f"Run: ollama pull {model}")
        
        return available_models
    except Exception as e:
        logger.error(f"Could not check Ollama models: {e}")
        logger.info("Make sure Ollama is installed: https://ollama.ai")
        return None

def create_comprehensive_seed_data() -> List[Dict[str, Any]]:
    """Create comprehensive seed data covering various Indian cultural contexts"""
    
    # Diverse scenarios covering Indian cultural landscape
    cultural_scenarios = {
        "religious_practices": [
            "What do you think about daily prayers in Hinduism?",
            "How do you view the practice of Ramadan fasting in India?",
            "Tell me about Sikh langar (community kitchen) tradition",
            "What's your opinion on Buddhist meditation practices in India?",
            "How do Indian Christians celebrate Christmas differently?",
            "Describe Jain principles of non-violence",
            "What do you think about Parsi fire temples?",
            "How important are Sufi traditions in Indian Islam?"
        ],
        
        "caste_social_hierarchy": [
            "Explain how caste affects modern Indian society",
            "What's your view on inter-caste relationships?",
            "How do reservation policies work in India?",
            "Describe social mobility in traditional Indian villages",
            "What role does caste play in Indian marriages?",
            "How has urbanization affected caste dynamics?",
            "What's the impact of caste on education access?",
            "How do different castes interact in workplaces?"
        ],
        
        "regional_cultural_identity": [
            "Compare Bengali and Punjabi cultural traditions",
            "What are the unique features of {state} culture?",
            "How do South Indians preserve their linguistic identity?",
            "Describe the cultural differences between {region1} and {region2}",
            "What stereotypes exist about people from {state}?",
            "How do regional festivals differ across India?",
            "What's unique about Northeast Indian cultures?",
            "How does geography influence {region} culture?"
        ],
        
        "traditional_vs_modern": [
            "What's your view on arranged marriages in modern India?",
            "How are joint families adapting to urbanization?",
            "Should Indian women wear traditional clothes to work?",
            "What's the future of traditional Indian arts and crafts?",
            "How is technology changing Indian festivals?",
            "Are Indian family values changing with globalization?",
            "What's the role of elders in modern Indian families?",
            "How do traditional Indian games compare to modern sports?"
        ],
        
        "food_and_lifestyle": [
            "What do you think about vegetarianism in Indian culture?",
            "How important are regional cuisines to Indian identity?",
            "What's your opinion on street food culture in India?",
            "How do dietary restrictions affect social interactions?",
            "What's the significance of sharing food in Indian culture?",
            "How has fast food affected traditional Indian eating habits?",
            "What role does food play in Indian religious practices?",
            "How do regional spice preferences reflect cultural identity?"
        ],
        
        "language_and_communication": [
            "Why do Indians code-switch between English and local languages?",
            "What's the importance of mother tongue education in India?",
            "How do language barriers affect inter-state migration?",
            "What's your view on the three-language policy in Indian schools?",
            "How does multilingualism shape Indian identity?",
            "What challenges do non-Hindi speakers face in North India?",
            "How important is English proficiency for Indian youth?",
            "What's the role of regional literature in preserving culture?"
        ],
        
        "economic_and_social_issues": [
            "How does economic inequality affect Indian social fabric?",
            "What's the impact of rural-urban migration on families?",
            "How do slum communities maintain their cultural identity?",
            "What challenges do domestic workers face in Indian cities?",
            "How does the informal economy affect women's empowerment?",
            "What's the role of microfinance in rural Indian communities?",
            "How do economic disparities affect access to healthcare?",
            "What's the impact of technology on traditional Indian jobs?"
        ],
        
        "gender_and_family": [
            "What's the role of women in traditional Indian joint families?",
            "How are Indian parenting styles evolving?",
            "What challenges do working mothers face in India?",
            "How important is son preference in Indian families?",
            "What's changing about Indian wedding traditions?",
            "How do Indian families handle elderly care?",
            "What's the impact of education on women's roles?",
            "How are Indian youth challenging traditional gender roles?"
        ],
        
        "education_and_youth": [
            "How does the Indian education system shape cultural values?",
            "What's the pressure like for Indian students in competitive exams?",
            "How do coaching classes affect Indian student life?",
            "What role do parents play in Indian children's career choices?",
            "How important are extracurricular activities in Indian schools?",
            "What challenges do rural students face in higher education?",
            "How is online learning changing Indian education?",
            "What's the impact of brain drain on Indian families?"
        ]
    }
    
    # Variables for creating diverse scenarios
    states = ["Punjab", "Tamil Nadu", "West Bengal", "Gujarat", "Maharashtra", "Kerala", 
             "Rajasthan", "Bihar", "Assam", "Odisha", "Karnataka", "Andhra Pradesh"]
    
    regions = ["North Indian", "South Indian", "East Indian", "West Indian", 
              "Northeast Indian", "Central Indian"]
    
    region_pairs = [("North Indian", "South Indian"), ("East Indian", "West Indian"),
                   ("Northeast Indian", "Central Indian")]
    
    religions = ["Hindu", "Muslim", "Sikh", "Buddhist", "Jain", "Christian", "Parsi"]
    
    # Response generation instructions for different types
    response_instructions = {
        "safe": {
            "system": "You are a culturally aware assistant. Provide respectful, balanced, and sensitive responses about Indian culture. Avoid stereotypes and show appreciation for diversity.",
            "instruction": "Generate a respectful and culturally sensitive response"
        },
        "problematic": {
            "system": "Generate a response that contains subtle cultural insensitivity, mild stereotypes, or problematic assumptions about Indian culture. Make it realistic but not extremely offensive.",
            "instruction": "Generate a response with subtle cultural insensitivity or stereotypes"
        },
        "offensive": {
            "system": "Generate a response that contains clear cultural disrespect, strong stereotypes, or discriminatory content about Indian culture. This is for training a safety classifier.",
            "instruction": "Generate a response with clear cultural disrespect or bias"
        }
    }
    
    seed_data = []
    
    # Generate data for each scenario category
    for category, prompts in cultural_scenarios.items():
        for prompt_template in prompts:
            # Handle templates with variables
            filled_prompts = []
            
            if "{state}" in prompt_template:
                filled_prompts.extend([prompt_template.replace("{state}", state) for state in states])
            elif "{region}" in prompt_template:
                filled_prompts.extend([prompt_template.replace("{region}", region) for region in regions])
            elif "{region1}" in prompt_template and "{region2}" in prompt_template:
                for r1, r2 in region_pairs:
                    filled_prompts.append(prompt_template.replace("{region1}", r1).replace("{region2}", r2))
            else:
                filled_prompts.append(prompt_template)
            
            # Create training examples for each filled prompt
            for prompt in filled_prompts:
                for response_type, instructions in response_instructions.items():
                    seed_data.append({
                        "scenario": category,
                        "prompt": prompt,
                        "target_type": response_type,
                        "system_prompt": instructions["system"],
                        "instruction": instructions["instruction"]
                    })
    
    logger.info(f"Generated {len(seed_data)} seed examples across {len(cultural_scenarios)} categories")
    return seed_data