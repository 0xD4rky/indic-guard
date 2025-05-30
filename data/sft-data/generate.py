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

