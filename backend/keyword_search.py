import chromadb
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_collection(
    name="ai_documents"
)

data = collection.get()

documents = data["documents"]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

question = input("Enter your question: ")

tokenized_question = question.lower().split()

scores = bm25.get_scores(tokenized_question)

ranked = sorted(
    zip(scores, documents),
    reverse=True
)

for i, (score, document) in enumerate(ranked[:3], start=1):

    print(f"\n--- BM25 RESULT {i} ---")
    print(f"Score: {score:.4f}")
    print("\nText:")
    print(document)