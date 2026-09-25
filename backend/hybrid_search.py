import ollama
import chromadb
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_collection(
    name="ai_documents"
)

data = collection.get()

documents = data["documents"]
ids = data["ids"]

question = input("Enter your question: ")

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]

vector_results = collection.query(
    query_embeddings=[question_embedding],
    n_results=len(documents)
)

vector_ids = vector_results["ids"][0]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

tokenized_question = question.lower().split()

bm25_scores = bm25.get_scores(tokenized_question)

bm25_ranked = sorted(
    zip(bm25_scores, ids),
    reverse=True
)

bm25_ids = [
    item[1]
    for item in bm25_ranked
]

rrf_scores = {}

k = 60

for rank, doc_id in enumerate(vector_ids, start=1):
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

for rank, doc_id in enumerate(bm25_ids, start=1):
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

ranked_ids = sorted(
    rrf_scores,
    key=rrf_scores.get,
    reverse=True
)

document_map = dict(zip(ids, documents))

for i, doc_id in enumerate(ranked_ids[:3], start=1):

    print(f"\n--- HYBRID RESULT {i} ---")
    print(f"Document ID: {doc_id}")
    print(f"RRF Score: {rrf_scores[doc_id]:.6f}")
    print("\nText:")
    print(document_map[doc_id])