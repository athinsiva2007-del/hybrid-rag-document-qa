import fitz
import re
import ollama
import chromadb

pdf_path = "data/documents/AI_AGI_ASI_Simple_Notes.pdf"

doc = fitz.open(pdf_path)

full_text = ""

for page in doc:
    full_text += page.get_text() + "\n"

doc.close()

paragraphs = re.split(r"\n\s*\n", full_text)

chunks = []
current_chunk = ""

for paragraph in paragraphs:
    paragraph = paragraph.strip()

    if not paragraph:
        continue

    if len(current_chunk) + len(paragraph) <= 800:
        current_chunk += paragraph + "\n"
    else:
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        current_chunk = paragraph + "\n"

if current_chunk.strip():
    chunks.append(current_chunk.strip())

print(f"Created {len(chunks)} chunks")

client = chromadb.PersistentClient(path="vectorstore")

try:
    client.delete_collection("ai_documents")
except:
    pass

collection = client.create_collection(
    name="ai_documents"
)

stored = 0

for i, chunk in enumerate(chunks):

    try:
        response = ollama.embed(
            model="nomic-embed-text",
            input=chunk
        )

        if not response.get("embeddings"):
            print(f"Chunk {i + 1} returned no embedding")
            continue

        embedding = response["embeddings"][0]

        collection.add(
            ids=[f"chunk_{i + 1}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": "AI_AGI_ASI_Simple_Notes.pdf",
                "chunk": i + 1
            }]
        )

        stored += 1

    except Exception as e:
        print(f"Chunk {i + 1} failed: {e}")

print(f"Stored {stored} embeddings in ChromaDB")