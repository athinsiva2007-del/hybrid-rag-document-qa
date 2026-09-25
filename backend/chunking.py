import fitz
import re

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
        chunks.append(current_chunk.strip())
        current_chunk = paragraph + "\n"

if current_chunk:
    chunks.append(current_chunk.strip())

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)