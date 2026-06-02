import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Choose a model
model_name = "gpt2"

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Device:", device)

# 4. Input text
text = "A transformer language model is"

# 5. Convert text into token IDs
inputs = tokenizer(text, return_tensors="pt").to(device)

print("\nOriginal text:")
print(text)

print("\nToken IDs:")
print(inputs["input_ids"])

# 6. Convert token IDs back into readable tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("\nTokens:")
print(tokens)

# 7. Look at GPT-2's token embedding layer
embedding_layer = model.transformer.wte

print("\nEmbedding layer:")
print(embedding_layer)

# 8. Convert token IDs into embeddings
token_embeddings = embedding_layer(inputs["input_ids"])

print("\nToken embedding shape")
print(token_embeddings.shape)