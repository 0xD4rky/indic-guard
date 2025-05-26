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
import logging
from typing import List, Dict, Any

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, GroupColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks.base import Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "total_samples": 5000,
    "batch_size": 50,
    "model": "gpt-4-turbo-preview",  
    "temperature": 0.7,
    "max_tokens": 300
}