import ollama

text = "Artificial Intelligence is the simulation of human intelligence by machines."

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print("Text:")
print(text)

print("\nEmbedding:")
print(embedding)

print("\nVector length:")
print(len(embedding))
