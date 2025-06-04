from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch

"""
1B -> 2GB of memory (inference)
"""

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    low_cpu_mem_usage=True,
    torch_dtype = torch.float16
).to(device)

model.eval()
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Why are Indians so poor?"},]
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=512)

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(outputs)