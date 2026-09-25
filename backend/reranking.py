import ollama
import chromadb
from sentence_transformers import CrossEncoder

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
    n_results=5
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

pairs = []

for document in documents:
    pairs.append([question, document])

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

scores = model.predict(pairs)

ranked = sorted(
    zip(scores, documents, metadatas),
    reverse=True
)

for i, (score, document, metadata) in enumerate(ranked, start=1):

    print(f"\n--- RERANKED RESULT {i} ---")
    print(f"Score: {score:.4f}")
    print(f"Metadata: {metadata}")
    print("\nText:")
    print(document)