import torch
from transformers import AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentences = [
    "A transformer language model is",
    "A rabbit is eating hay",
    "The actor speaks with dramatic emotion"
]

for sentence in sentences:
    print("\n==============================")
    print("Original text:")
    print(sentence)

    inputs = tokenizer(sentence, return_tensors="pt")

    print("\nToken IDs:")
    print(inputs["input_ids"])

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print("\nTokens:")
    print(tokens)

    print("\nNumber of tokens:")
    print(len(tokens))
