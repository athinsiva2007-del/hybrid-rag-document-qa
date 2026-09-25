import ollama
import chromadb

client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_collection(
    name="ai_documents"
)

question = "What is Artificial General Intelligence?"

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

documents = results["documents"][0]
distances = results["distances"][0]
metadatas = results["metadatas"][0]

for i, document in enumerate(documents):
    print(f"\n--- RESULT {i + 1} ---")
    print(f"Distance: {distances[i]}")
    print(f"Metadata: {metadatas[i]}")
    print("\nText:")
    print(document)