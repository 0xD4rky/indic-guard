import os
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, TextGeneration, KeepColumns
from distilabel.llms import vLLM
import json
import random

# my cases:
INDIAN_CULTURAL_CATEGORIES = {
    "RELIGION": ["Hindu", "Muslim", "Sikh", "Christian", "Buddhist", "Jain", "Parsi"],
    "REGIONS": ["North India", "South India", "East India", "West India", "Northeast India"],
    "LANGUAGES": ["Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"],
    "CASTES": ["Brahmin", "Kshatriya", "Vaishya", "Shudra", "Dalit communities"],
    "FESTIVALS": ["Diwali", "Holi", "Eid", "Dussehra", "Ganesh Chaturthi", "Durga Puja", "Onam"],
    "FOOD": ["vegetarian traditions", "non-vegetarian food", "regional cuisines", "food taboos"],
    "TRADITIONS": ["arranged marriages", "joint family system", "clothing traditions", "cultural ceremonies"]
}
