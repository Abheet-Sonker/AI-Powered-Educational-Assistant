# ✅ Updated ingest.py with improved chunking and metadata tagging
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json
from langchain.docstore.document import Document
import os

# 1. Load transcript file (assuming already cleaned)
loader = TextLoader("transcript_file.txt", encoding="utf-8")
documents = loader.load()

# 2. Smarter chunking (larger chunks, slight overlap)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1400,       # to retain formulas and full examples
    chunk_overlap=200,
    separators=["\n\n", "\n"]  # prioritize breaking on paragraphs or sentences
)
chunks = text_splitter.split_documents(documents)

# Print sample chunks for review
for i, doc in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:500]}")

# 3. Embedding model (community, free)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Build vector DB and save
vector_db = FAISS.from_documents(chunks, embedding)
vector_db.save_local("faiss_index")

print("✅ Ingestion complete. New vector DB saved!")