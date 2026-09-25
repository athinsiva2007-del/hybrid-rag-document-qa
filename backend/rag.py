import ollama
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_collection(
    name="ai_documents"
)

data = collection.get()

documents = data["documents"]
ids = data["ids"]
metadatas = data["metadatas"]

question = input("Ask a question: ")

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

if not response.get("embeddings"):
    print("Error: Could not create question embedding.")
    exit()

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
    rrf_scores[doc_id] = (
        rrf_scores.get(doc_id, 0)
        + 1 / (k + rank)
    )

for rank, doc_id in enumerate(bm25_ids, start=1):
    rrf_scores[doc_id] = (
        rrf_scores.get(doc_id, 0)
        + 1 / (k + rank)
    )

hybrid_ids = sorted(
    rrf_scores,
    key=rrf_scores.get,
    reverse=True
)

document_map = dict(zip(ids, documents))
metadata_map = dict(zip(ids, metadatas))

candidate_ids = hybrid_ids[:4]

candidate_documents = [
    document_map[doc_id]
    for doc_id in candidate_ids
]

pairs = [
    [question, document]
    for document in candidate_documents
]

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

scores = reranker.predict(pairs)

ranked = sorted(
    zip(
        scores,
        candidate_ids,
        candidate_documents
    ),
    reverse=True
)

top_results = ranked[:3]

context = ""

for i, (score, doc_id, document) in enumerate(
    top_results,
    start=1
):
    metadata = metadata_map[doc_id]

    context += f"""
SOURCE {i}
File: {metadata["source"]}
Chunk: {metadata["chunk"]}

Content:
{document}

"""

prompt = f"""
Answer the question using only the provided context.

If the context does not contain enough information, say:
"I don't have enough information in the provided document."

Do not create facts that are not present in the context.
Do not create or invent source names.

Context:
{context}

Question:
{question}

Answer:
"""

response = ollama.chat(
    model="qwen2.5:3b-instruct",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]

print("\n--- ANSWER ---")
print(answer)

print("\n--- SOURCES ---")

for i, (score, doc_id, document) in enumerate(
    top_results,
    start=1
):
    metadata = metadata_map[doc_id]

    print(
        f"[{i}] {metadata['source']} "
        f"(Chunk {metadata['chunk']})"
    )