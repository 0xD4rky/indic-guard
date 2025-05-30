import json
import re
from vllm import LLM, SamplingParams
from typing import List, Dict, Any
import random

# configs:

model_name = "meta-llama/Llama-3.1-8B-Instruct"
output_file = "indic_guard_sft_data_generated.jsonl"
examples_per_category = 50  
batch_size = 10




labels = {
    "SAFE": "Content that is respectful and appropriate within Indian cultural context",
    "HATE_SPEECH_RELIGION": "Hateful content targeting religious groups in India",
    "HATE_SPEECH_CASTE": "Hateful content targeting specific castes",
    "HATE_SPEECH_COMMUNITY_ETHNICITY": "Hateful content targeting Indian communities/ethnicities",
    "DISPARAGEMENT_TRADITION_CULTURE": "Content mocking or belittling Indian traditions/culture",
    "OFFENSIVE_STEREOTYPE_INDIAN": "Content perpetuating harmful stereotypes about Indians",
    "INCITEMENT_VIOLENCE_INDIAN_CONTEXT": "Content inciting violence in Indian social/political context",
    "RELIGIOUS_INSENSITIVITY_INDIAN": "Content disrespectful to Indian religious practices/figures",
    "CASTE_DISCRIMINATION_SLURS": "Content promoting caste discrimination or using casteist slurs",
    "MISINFO_DISINFO_CULTURALLY_HARMFUL_INDIAN": "False information harmful to Indian culture/society",
    "SEXUAL_HARASSMENT_INDIAN_CONTEXT": "Sexual content offensive to Indian social norms",
    "OTHER_CULTURALLY_OFFENSIVE_INDIAN": "Other culturally offensive content specific to India"
}


class Augmenter:

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.llm = None

    def initialize_llm(self):
        if self.llm is None:
            print("Initializing LLM for augmentation...")
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.9
            )
            print("LLM initialized.")

    def paraphrase_examples(self, examples: List[Dict], target_count: int = 5000) -> List[Dict]:
        self.initialize_llm()
        paraphrased_data = []
        sampling_params = SamplingParams(temperature=0.8, max_tokens=500, top_p=0.9)
        print(f"Generating paraphrases to reach {target_count} examples...")
        while len(paraphrased_data) < target_count:
            batch = random.sample(examples, min(10, len(examples)))
            for example in batch:
                if len(paraphrased_data) >= target_count:
                    break
                prompt = f"""Paraphrase the following text while keeping the same meaning and harmful/safe nature:

Original: "{example['text_to_classify']}"
Label: {example['label']}

Create 3 different paraphrases that:
1. Use different words but same meaning
2. Change sentence structure but preserve intent
3. Adjust tone (formal/informal) but keep the same sentiment

Return as JSON array: [{{"text_to_classify": "paraphrase1", "label": "{example['label']}"}}, ...]
"""
                try:
                    outputs = self.llm.generate([prompt], sampling_params)
                    response = outputs[0].outputs[0].text.strip()
                    response = re.sub(r'```json\s*', '', response)
                    response = re.sub(r'```\s*$', '', response)
                    parsed = json.loads(response)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "text_to_classify" in item:
                                paraphrased_data.append(item)
                                if len(paraphrased_data) >= target_count:
                                    break
                except Exception as e:
                    print(f"Paraphrase error: {e}")
                    continue
        return paraphrased_data[:target_count]

    def generate_contextual_variations(self, base_examples: List[Dict]) -> List[Dict]:
        self.initialize_llm()
        contexts = [
            "social media post", "formal email", "casual conversation", "news comment",
            "workplace discussion", "family conversation", "academic discussion", "online forum post"
        ]
        variations = []
        sampling_params = SamplingParams(temperature=0.9, max_tokens=600)
        for example in base_examples[:500]:
            context = random.choice(contexts)
            prompt = f"""Rewrite this text as it would appear in a {context}, while maintaining the same harmful/safe nature:

Original: "{example['text_to_classify']}"
Label: {example['label']}

Rewrite for context: {context}

Return JSON: {{"text_to_classify": "rewritten_text", "label": "{example['label']}"}}
"""
            try:
                outputs = self.llm.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text.strip()
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if "text_to_classify" in parsed:
                        variations.append(parsed)
            except:
                continue
        return variations

    def generate_edge_cases(self) -> List[Dict]:
        self.initialize_llm()
        edge_case_prompts = [
            "Generate examples that are borderline between SAFE and harmful categories",
            "Generate examples with mixed Hindi-English (Hinglish) content",
            "Generate examples with implicit bias rather than explicit hate",
            "Generate examples that use coded language or dog whistles",
            "Generate examples that seem positive but contain subtle discrimination",
        ]
        edge_cases = []
        sampling_params = SamplingParams(temperature=0.85, max_tokens=1000)
        for prompt_template in edge_case_prompts:
            full_prompt = f"""{prompt_template}

Create 20 examples as JSON array. Each should be challenging to classify.
Format: [{{"text_to_classify": "...", "label": "APPROPRIATE_LABEL"}}, ...]

Use the Indic-Guard taxonomy: SAFE, HATE_SPEECH_RELIGION, HATE_SPEECH_CASTE, 
HATE_SPEECH_COMMUNITY_ETHNICITY, DISPARAGEMENT_TRADITION_CULTURE, 
OFFENSIVE_STEREOTYPE_INDIAN, INCITEMENT_VIOLENCE_INDIAN_CONTEXT, 
RELIGIOUS_INSENSITIVITY_INDIAN, CASTE_DISCRIMINATION_SLURS, 
MISINFO_DISINFO_CULTURALLY_HARMFUL_INDIAN, SEXUAL_HARASSMENT_INDIAN_CONTEXT, 
OTHER_CULTURALLY_OFFENSIVE_INDIAN
"""
            try:
                outputs = self.llm.generate([full_prompt], sampling_params)
                response = outputs[0].outputs[0].text.strip()
                response = re.sub(r'```json\s*', '', response)
                response = re.sub(r'```\s*$', '', response)
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    edge_cases.extend(parsed)
            except Exception as e:
                print(f"Edge case generation error: {e}")
                continue
        return edge_cases

    def balance_dataset(self, data: List[Dict], target_safe_ratio: float = 0.4) -> List[Dict]:
        safe_examples = [ex for ex in data if ex['label'] == 'SAFE']
        harmful_examples = [ex for ex in data if ex['label'] != 'SAFE']
        target_safe_count = int(len(data) * target_safe_ratio)
        target_harmful_count = len(data) - target_safe_count
        if len(safe_examples) < target_safe_count:
            additional_safe = self.generate_additional_safe_examples(
                target_safe_count - len(safe_examples)
            )
            safe_examples.extend(additional_safe)
        else:
            safe_examples = random.sample(safe_examples, target_safe_count)
        if len(harmful_examples) > target_harmful_count:
            harmful_examples = random.sample(harmful_examples, target_harmful_count)
        balanced_data = safe_examples + harmful_examples
        random.shuffle(balanced_data)
        return balanced_data

    def generate_additional_safe_examples(self, count: int) -> List[Dict]:
        self.initialize_llm()
        safe_prompts = [
            "questions about Indian history and culture",
            "compliments about Indian food, art, or traditions",
            "requests for travel advice in India",
            "discussions about Indian festivals and celebrations",
            "appreciation for Indian literature and philosophy",
            "questions about learning Indian languages",
            "positive comments about Indian achievements",
            "respectful discussions about Indian diversity"
        ]
        safe_examples = []
        sampling_params = SamplingParams(temperature=0.8, max_tokens=800)
        examples_per_prompt = count // len(safe_prompts) + 1
        for topic in safe_prompts:
            prompt = f"""Generate {examples_per_prompt} examples of SAFE content about {topic}.

These should be:
- Respectful and positive
- Genuinely curious or appreciative
- Free from any bias or stereotypes
- Representative of how people positively engage with Indian culture

Return as JSON array: [{{"text_to_classify": "...", "label": "SAFE"}}, ...]
"""
            try:
                outputs = self.llm.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text.strip()
                response = re.sub(r'```json\s*', '', response)
                response = re.sub(r'```\s*$', '', response)
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    safe_examples.extend(parsed[:examples_per_prompt])
                if len(safe_examples) >= count:
                    break
            except:
                continue
        return safe_examples[:count]
    



def get_category_prompt(category: str, batch_num: int) -> str:
    base_instruction = f"""You are creating training data for an AI safety model called "Indic-Guard" that detects content harmful to Indian culture.

Generate exactly {batch_size} diverse examples for the category: {category}

CRITICAL REQUIREMENTS:
1. Create a JSON array with exactly {batch_size} objects
2. Each object must have: {{"text_to_classify": "...", "label": "{category}"}}
3. Make examples realistic - they should sound like real user messages or AI responses
4. Include both SUBTLE and OBVIOUS examples of the harmful category
5. Vary the length, tone, and context of examples
6. For harmful categories, create examples that someone might actually write

"""
    
    category_specific = {
        "SAFE": f"""Create {batch_size} examples of SAFE content in Indian context:
- Respectful questions about Indian culture, history, traditions
- Neutral factual statements about India
- Polite requests for information about Indian customs
- Appreciative comments about Indian arts, food, or festivals
- Educational content about Indian languages, geography, etc.

Examples should be diverse in topic and style.""",

        "HATE_SPEECH_RELIGION": f"""Create {batch_size} examples of religious hate speech in Indian context:
- Direct attacks on specific religions (Hindu, Muslim, Sikh, Christian, Buddhist, Jain, etc.)
- Calls for discrimination against religious groups
- Inflammatory generalizations about religious communities
- Content promoting religious supremacy or hatred
- Derogatory comments about religious practices or beliefs

Include both subtle dog-whistles and overt hate speech.""",

        "HATE_SPEECH_CASTE": f"""Create {batch_size} examples of caste-based hate speech:
- Derogatory references to specific castes (using actual caste names)
- Justification of caste-based discrimination
- Caste-based slurs and insults
- Content promoting caste superiority/inferiority
- Calls for violence against specific caste groups

Include both traditional and modern forms of casteism.""",

        "HATE_SPEECH_COMMUNITY_ETHNICITY": f"""Create {batch_size} examples targeting Indian ethnic/linguistic communities:
- Hate speech against regional groups (Tamils, Bengalis, Punjabis, etc.)
- Attacks on tribal communities
- Discrimination against linguistic minorities
- Ethnic slurs and stereotypes
- Calls for violence against specific communities

Focus on actual ethnic tensions that exist in India.""",

        "DISPARAGEMENT_TRADITION_CULTURE": f"""Create {batch_size} examples mocking Indian traditions:
- Calling Indian festivals "primitive" or "backward"
- Mocking traditional clothing, food, or customs
- Ridiculing Indian art forms, music, or dance
- Dismissing Indian philosophical concepts as "nonsense"
- Degrading traditional practices like arranged marriages, joint families
- Making fun of Indian languages or accents

Include both direct mockery and subtle condescension.""",

        "OFFENSIVE_STEREOTYPE_INDIAN": f"""Create {batch_size} examples of harmful Indian stereotypes:
- Professional stereotypes (tech support, convenience stores, etc.)
- Physical appearance stereotypes
- Character trait stereotypes (cheap, smelly, etc.)
- Regional stereotypes (all South Indians are dark, all Punjabis are loud, etc.)
- Cultural practice stereotypes
- Economic status stereotypes

Make them sound like real prejudiced comments people might make.""",

        "INCITEMENT_VIOLENCE_INDIAN_CONTEXT": f"""Create {batch_size} examples of violence incitement in Indian context:
- Calls for riots against specific communities
- Threats related to political tensions
- Incitement around sensitive historical events
- Calls for violence over religious/cultural issues
- Threats against specific ethnic or caste groups

Be realistic about actual sources of tension in Indian society.""",

        "RELIGIOUS_INSENSITIVITY_INDIAN": f"""Create {batch_size} examples of religious insensitivity:
- Vulgar jokes about Hindu deities
- Offensive comments about Islamic practices
- Disrespectful remarks about Sikh traditions
- Crude references to Christian or Buddhist practices
- Trivializing sacred texts or religious symbols
- Inappropriate sexualization of religious figures

Include subtle insensitivity, not just obvious blasphemy.""",

        "CASTE_DISCRIMINATION_SLURS": f"""Create {batch_size} examples of caste discrimination:
- Usage of actual casteist slurs (but in educational context)
- Discriminatory statements about "untouchability"
- Caste-based job discrimination
- Marriage-related caste prejudice
- Educational or social exclusion based on caste
- Modern forms of caste discrimination (online, workplace, etc.)

Include both traditional and contemporary forms.""",

        "MISINFO_DISINFO_CULTURALLY_HARMFUL_INDIAN": f"""Create {batch_size} examples of culturally harmful misinformation:
- False claims about Indian history designed to create division
- Fake statistics to promote prejudice
- Conspiracy theories targeting specific communities
- False religious or cultural "facts" meant to cause offense
- Manipulated historical narratives
- Health misinformation targeting cultural practices

Make them believable but clearly false.""",

        "SEXUAL_HARASSMENT_INDIAN_CONTEXT": f"""Create {batch_size} examples of sexual harassment in Indian context:
- Objectifying comments about Indian women's clothing choices
- Sexual remarks about cultural practices or festivals
- Inappropriate comments about arranged marriages
- Harassment related to Indian cultural norms around modesty
- Sexual stereotypes about Indian women
- Harassment that specifically violates Indian cultural sensitivities

Focus on content that's particularly offensive given Indian social norms.""",

        "OTHER_CULTURALLY_OFFENSIVE_INDIAN": f"""Create {batch_size} examples of other cultural offenses:
- Extreme disrespect to elders in culturally specific ways
- Violations of Indian hospitality norms
- Offensive behavior around food customs
- Disrespect for Indian family structures
- Inappropriate behavior around cultural ceremonies
- Violations of respect norms specific to Indian culture

Include subtle cultural insensitivity that Indians would find offensive."""
    }

    return base_instruction + category_specific.get(category, "Generate appropriate examples for this category.")


def clean_llm_response(text: str) -> str:
    """cleaning the llm response to extract json"""
    
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return text.strip()

def validate_and_fix_examples(examples: List[Dict], expected_label: str) -> List[Dict]:
    """validating and fixing generated examples"""
    valid_examples = []
    
    for example in examples:
        if not isinstance(example, dict):
            continue
            
        if "text_to_classify" not in example or "label" not in example:
            continue
        
        if example["label"] != expected_label:
            example["label"] = expected_label # fixing the label 'if' its incorrect
        
        if not example["text_to_classify"].strip():
            continue

        text = example["text_to_classify"]
        if len(text) < 10 or len(text) > 500:  # Reasonable length bounds
            continue
            
        valid_examples.append(example)
    
    return valid_examples


def generate_sft_data():
    """
    this func will be used for generating data for SFT using vllm
    """

    print(f"Initializing LLM: {model_name}")
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096, 
        gpu_memory_utilization=0.85 
    )
    print("LLM initialized successfully.")

    sampling_params = SamplingParams(
        temperature=0.8,  
        top_p=0.95,
        max_tokens=3000,
        frequency_penalty=0.2, 
        presence_penalty=0.1
    )

    all_generated_data = []
    total_categories = len(labels)
    
    print(f"\nGenerating {examples_per_category} examples per category across {total_categories} categories")
    print(f"Total expected examples: {total_categories * examples_per_category}")
    
    for category_idx, category in enumerate(labels.keys(), 1):
        print(f"\n=== Category {category_idx}/{total_categories}: {category} ===")
        
        category_examples = []
        batches_needed = (examples_per_category + batch_size - 1) // batch_size
        
        for batch_num in range(batches_needed):
            remaining = min(batch_size, examples_per_category - len(category_examples))
            if remaining <= 0:
                break
                
            print(f"  Generating batch {batch_num + 1}/{batches_needed} ({remaining} examples)")
            
            prompt = get_category_prompt(category, batch_num)
            
            try:
                outputs = llm.generate([prompt], sampling_params)
                raw_response = outputs[0].outputs[0].text
                
                cleaned_response = clean_llm_response(raw_response)
                
                try:
                    parsed_examples = json.loads(cleaned_response)
                    
                    if isinstance(parsed_examples, list):
                        valid_examples = validate_and_fix_examples(parsed_examples, category)
                        category_examples.extend(valid_examples[:remaining])
                        print(f"    Added {len(valid_examples[:remaining])} valid examples")
                    else:
                        print(f"    Warning: Response was not a list")
                        
                except json.JSONDecodeError as e:
                    print(f"    Error: Failed to parse JSON - {e}")
                    print(f"    Raw response (first 200 chars): {raw_response[:200]}...")
                    
            except Exception as e:
                print(f"    Critical Error: {e}")
                continue
        
        all_generated_data.extend(category_examples)
        print(f"  Total examples for {category}: {len(category_examples)}")
    
    random.shuffle(all_generated_data)
    
    print(f"\n=== Saving {len(all_generated_data)} examples to {output_file} ===")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_generated_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Generate statistics
    label_counts = {}
    for entry in all_generated_data:
        label = entry['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # stats
    print(f"Total examples generated: {len(all_generated_data)}")
    print("\nExamples per category:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    
    return output_file

if __name__ == '__main__':
    output_file = generate_sft_data()
    print(f"\nData generation complete! Check {output_file}")