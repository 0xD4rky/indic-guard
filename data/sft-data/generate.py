import os
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import vLLM
import json
import random

categories = {  
    "RELIGION": ["Hindu", "Muslim", "Sikh", "Christian", "Buddhist", "Jain", "Parsi"],
    "REGIONS": ["North India", "South India", "East India", "West India", "Northeast India"],
    "LANGUAGES": ["Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"],
    "CASTES": ["Brahmin", "Kshatriya", "Vaishya", "Shudra", "Dalit communities"],
    "FESTIVALS": ["Diwali", "Holi", "Eid", "Dussehra", "Ganesh Chaturthi", "Durga Puja", "Onam"],
    "FOOD": ["vegetarian traditions", "non-vegetarian food", "regional cuisines", "food taboos"],
    "TRADITIONS": ["arranged marriages", "joint family system", "clothing traditions", "cultural ceremonies"]
}

def generate_base_prompts():
    prompts = []
    
    # normal prompts
    normal_prompts = [
        "Can you tell me about {topic} in Indian culture?",
        "I'm interested in learning about {topic}. Can you explain?",
        "What is the significance of {topic} in Indian tradition?",
        "How do different regions of India celebrate {topic}?",
        "I want to understand the cultural importance of {topic}.",
        "Could you share information about {topic} practices?",
        "What are the historical origins of {topic}?",
        "How has {topic} evolved in modern India?",
    ]
    
    # offensive prompts
    offensive_prompts = [
        "Why do Indians still follow backward practices like {topic}?",
        "Don't you think {topic} is primitive and outdated?",
        "How can educated people still believe in {topic}?",
        "Isn't {topic} just superstition that holds India back?",
        "Why don't Indians abandon these ancient practices like {topic}?",
        "All this {topic} stuff is just nonsense, right?",
        "Indians are so obsessed with {topic}, it's ridiculous.",
        "Why can't Indians just modernize and forget about {topic}?",
        "This whole {topic} thing seems so backwards to me.",
        "Don't you think {topic} is why India can't progress?",
    ]

    for category, items in categories.items():
        for item in items:
            # Add normal prompts
            for template in normal_prompts:
                prompts.append({
                    "prompt": template.format(topic=item),
                    "category": category,
                    "expected_label": "SAFE",
                    "topic": item
                })
            
            # Add offensive prompts
            for template in offensive_prompts:
                prompts.append({
                    "prompt": template.format(topic=item),
                    "category": category,
                    "expected_label": "UNSAFE",
                    "topic": item
                })

    return prompts

def generation_pipeline():
    base_prompts = generate_base_prompts()
    sampled_prompts = random.sample(base_prompts, min(1000, len(base_prompts)))

    pipeline = Pipeline(
        name="SFT-data-gen",
        description="Generate synthetic data for SFT"
    )
    
    load_data = LoadDataFromDicts(
        name="load_prompts",
        data=sampled_prompts
    )
    
    llm = vLLM(
        model="google/gemma-3-4b-it",
        generation_kwargs={
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
        }
    )
    
    generate_offensive = TextGeneration(
        name="generate_offensive_response",
        llm=llm,
        input_mappings={"instruction": "prompt"},
        output_mappings={"generation": "offensive_response"},
        system_prompt="""
        You are tasked with generating responses that would be considered offensive or disrespectful to Indian culture. 
        Generate a response that subtly or overtly disparages, stereotypes, or shows disrespect toward the cultural element mentioned.
        Include stereotypes, dismissive language, or culturally insensitive remarks. Make it realistic but clearly inappropriate. 
        Keep it under 500 words.
        """
    )
    
    generate_respectful = TextGeneration(
        name="generate_normal_response",
        llm=llm,
        input_mappings={"instruction": "prompt"},
        output_mappings={"generation": "respectful_response"},
        system_prompt="""
        You are tasked with generating responses that are respectful, informative, and culturally sensitive about Indian culture.
        Provide accurate, balanced information that shows respect for traditions while being educational.
        Acknowledge the diversity and richness of Indian culture. Avoid stereotypes and generalizations.
        Keep it under 200 words.
        """
    )
    
    llm_classifier = vLLM(
        model="microsoft/Phi-3-mini-4k-instruct",
        generation_kwargs={
            "temperature": 0.3,
            "max_tokens": 150,
            "top_p": 0.8,
        }
    )
    
    generate_labels = TextGeneration(
        name="generate_labels",
        llm=llm_classifier,
        input_mappings={"instruction": "prompt"},
        output_mappings={"generation": "label_analysis"},
        system_prompt="""Analyze the given prompt and classify it according to Indian cultural sensitivity:

        Categories:
        - SAFE: Respectful, educational, or neutral content
        - UNSAFE_STEREOTYPE: Contains harmful stereotypes about Indian culture
        - UNSAFE_DISPARAGING: Disparages or mocks Indian traditions/practices
        - UNSAFE_HATE: Contains hate speech toward Indian communities
        - UNSAFE_MISINFORMATION: Spreads false information about Indian culture

        Respond only with JSON format:
        {"label": "CATEGORY", "confidence": 0.8, "reasoning": "brief explanation"}"""
    )
    
    # connecting all the steps
    load_data >> generate_offensive
    load_data >> generate_respectful
    load_data >> generate_labels
    
    return pipeline

def create_sft_dataset(pipeline_output):
    sft_data = []
    
    for row in pipeline_output:
        prompt = row["prompt"]
        expected_label = row.get("expected_label", "SAFE")
        category = row.get("category", "GENERAL")
        
        # determining whether the prompt is offensive or not
        offensive_keywords = ["backward", "primitive", "outdated", "superstition", "nonsense", "ridiculous", "obsessed"]
        is_offensive_prompt = any(keyword in prompt.lower() for keyword in offensive_keywords)
        
        if is_offensive_prompt:
            # for offensive prompts, label UNSAFE
            reasoning = row.get("offensive_response", "This prompt contains language that disparages or shows disrespect toward Indian cultural practices.")
            label = "UNSAFE_DISPARAGING"
        else:
            # for normal prompts, label SAFE
            reasoning = row.get("respectful_response", "This prompt shows respect and genuine interest in learning about Indian culture.")
            label = "SAFE"
        
        sft_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"{label}\n\nReasoning: {reasoning[:450]}..."}
            ],
            "label": label,
            "category": category,
            "topic": row.get("topic", ""),
        })
        
        if "offensive_response" in row and "respectful_response" in row:
            sft_data.append({
                "messages": [
                    {"role": "user", "content": f"Classify this response about Indian culture: {row['offensive_response'][:100]}..."},
                    {"role": "assistant", "content": "UNSAFE_DISPARAGING\n\nReasoning: This response contains stereotypes and dismissive language toward Indian cultural practices."}
                ],
                "label": "UNSAFE_DISPARAGING",
                "category": category,
                "topic": row.get("topic", ""),
            })
    
    return sft_data


def generate_fallback_dataset(target_size=1000):
    """
    Generate a fallback dataset when the main pipeline fails
    """
    base_prompts = generate_base_prompts()
    sft_data = []
    
    # Take a subset of prompts
    selected_prompts = random.sample(base_prompts, min(target_size // 2, len(base_prompts)))
    
    for prompt_data in selected_prompts:
        prompt = prompt_data["prompt"]
        expected_label = prompt_data["expected_label"]
        category = prompt_data["category"]
        topic = prompt_data["topic"]
        
        # Generate basic responses based on expected label
        if expected_label == "UNSAFE":
            response = f"UNSAFE_DISPARAGING\n\nReasoning: This prompt contains language that shows disrespect or uses negative framing toward {topic} in Indian culture."
        else:
            response = f"SAFE\n\nReasoning: This prompt shows genuine interest in learning about {topic} in Indian culture in a respectful manner."
        
        sft_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ],
            "label": expected_label,
            "category": category,
            "topic": topic
        })
    
    # Augment to reach target size
    while len(sft_data) < target_size:
        base_sample = random.choice(sft_data[:len(selected_prompts)])
        
        # Create variations
        original_prompt = base_sample["messages"][0]["content"]
        variations = [
            original_prompt.replace("Indian", "South Asian"),
            original_prompt.replace("culture", "traditions"),
            original_prompt.replace("practices", "customs"),
        ]
        
        for variation in variations:
            if len(sft_data) >= target_size:
                break
                
            sft_data.append({
                "messages": [
                    {"role": "user", "content": variation},
                    base_sample["messages"][1]
                ],
                "label": base_sample["label"],
                "category": base_sample["category"],
                "topic": base_sample["topic"]
            })
    
    return sft_data[:target_size]

def main(stats: bool):
    print("Starting SFT Data gen process")
    sft_dataset = []
    
    try:
        print("Attempting to use vLLM with gemma 3 4B")
        pipeline = generation_pipeline()
        
        result = pipeline.run(
            parameters={
                "load_prompts": {"batch_size": 32},
                "generate_offensive_response": {"generation_kwargs": {"max_tokens": 500, "temperature": 0.7}},
                "generate_respectful_response": {"generation_kwargs": {"max_tokens": 500, "temperature": 0.7}},
                "generate_labels": {"generation_kwargs": {"max_tokens": 100, "temperature": 0.3}},
            }
        )
        
        sft_dataset = create_sft_dataset(result)
        
    except Exception as e:
        print(f"vLLM approach failed: {e}")
        print("Falling back to basic dataset generation")
        sft_dataset = generate_fallback_dataset(1000)
    
    # Fixed: Handle empty dataset case
    if not sft_dataset:
        print("No data generated, creating fallback dataset")
        sft_dataset = generate_fallback_dataset(1000)
    
    
    
    output_file = "sft_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(sft_dataset)} SFT examples")
    print(f"Saved to '{output_file}'")
    
    if stats:
        label_counts = {}
        category_counts = {}
        
        for sample in sft_dataset:
            label = sample["label"]
            category = sample["category"]
            
            label_counts[label] = label_counts.get(label, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\nDataset statistics:")
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} examples")
        
        print("\nCategory distribution:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} examples")
    
    # looking at a sample
    print("\nSample SFT data:")
    for i, sample in enumerate(sft_dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"User: {sample['messages'][0]['content']}")
        print(f"Assistant: {sample['messages'][1]['content'][:100]}...")
        print(f"Label: {sample['label']}")
        print(f"Category: {sample['category']}")

if __name__ == "__main__":
    main(True)