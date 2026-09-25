import fitz

pdf_path = "data/documents/AI_AGI_ASI_Simple_Notes.pdf"

doc = fitz.open(pdf_path)

for page_number, page in enumerate(doc, start=1):
    text = page.get_text()

    print(f"\n--- PAGE {page_number} ---")
    print(text)

doc.close()