import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

text = "A transformer language model is"

inputs = tokenizer(text, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

print("Input IDs:")
print(input_ids)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("\nTokens:")
print(tokens)

# Token embedding layer: tells the model what each token is
token_embedding_layer = model.transformer.wte

# Position embedding layer: tells the model where each token is
position_embedding_layer = model.transformer.wpe

print("\nToken embedding layer:")
print(token_embedding_layer)

print("\nPosition embedding layer:")
print(position_embedding_layer)

# Create position IDs: 0, 1, 2, 3, ...
sequence_length = input_ids.shape[1]
position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)

print("\nPosition IDs:")
print(position_ids)

# Convert token IDs into token embeddings
token_embeddings = token_embedding_layer(input_ids)

# Convert position IDs into position embeddings
position_embeddings = position_embedding_layer(position_ids)

print("\nToken embeddings shape:")
print(token_embeddings.shape)

print("\nPosition embeddings shape:")
print(position_embeddings.shape)

# GPT-2 input to Transformer blocks
combined_embeddings = token_embeddings + position_embeddings

print("\nCombined embeddings shape:")
print(combined_embeddings.shape)