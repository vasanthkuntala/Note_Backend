import os
import uuid
import fitz  # PyMuPDF
import time
import requests
import asyncio
import numpy as np
from typing import List
from fastapi import  File, UploadFile,HTTPException

from pinecone import Pinecone, ServerlessSpec

PINECONE_INDEX_NAME = "gemini-embeddings-index"
PINECONE_API_KEY = "PINECONE_API_KEY"
GEMINI_API_KEY = "GEMINI_API_KEY"
UPLOAD_DIR = "uploaded_pdfs"
EMBEDDING_DIM = 768

os.makedirs(UPLOAD_DIR, exist_ok=True)
pinecone = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

# -------- Gemini Embedding Function --------
def get_gemini_embeddings(api_key: str, text: str, max_retries: int = 5, base_delay: int = 1):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/gemini-embedding-exp-03-07",
        "content": {"parts": [{"text": text}]}
    }
    for retries in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["embedding"]["values"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                delay = base_delay * (2 ** retries)
                print(f"⏳ Rate limited. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"❌ HTTP error: {e}")
                break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
    return None

# -------- PDF Text Extraction --------
def extract_text_with_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text().strip() + "\n"
    doc.close()
    return full_text

# -------- Chunking Function --------
def chunk_text_fixed_words(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


async def process_uploaded_files(files: List[UploadFile] = File(...)):
    results = []
    chunk_counter = 0
    upsert_data = []

    saved_file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_file_paths.append(file_path)

    for file_path in saved_file_paths:
        filename = os.path.basename(file_path)
        try:
            print(f"🔍 Processing: {filename}")

            raw_text = extract_text_with_pymupdf(file_path)
            chunks = chunk_text_fixed_words(raw_text, chunk_size=500, overlap=50)

            for chunk in chunks:
                embedding_values = await asyncio.get_event_loop().run_in_executor(
                    None, get_gemini_embeddings, GEMINI_API_KEY, chunk
                )
                if embedding_values:
                    vector = np.array(embedding_values, dtype=np.float32).tolist()
                    chunk_id = str(chunk_counter)
                    chunk_counter += 1
                    upsert_data.append((chunk_id, vector, {"text": chunk}))
  # No metadata
                else:
                    print(f"⚠️ Failed to embed chunk from {filename}")

            results.append({"file": filename, "status": "Success"})
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            results.append({"file": filename, "status": "Failed", "error": str(e)})

    if upsert_data:
        print(f"⬆️ Upserting {len(upsert_data)} chunks to Pinecone...")
        pinecone_index.upsert(vectors=upsert_data)
        print("✅ Upsert complete.")

    return results





async def query_faiss_index(query: str):
   
    embedding_values = await asyncio.get_event_loop().run_in_executor(
        None, get_gemini_embeddings, GEMINI_API_KEY, query
    )
    if not embedding_values:
        raise HTTPException(status_code=500, detail="❌ Failed to generate query embeddings")
    query_vector = np.array(embedding_values, dtype=np.float32).tolist()

   
 
    k = 5
   
    query_results = pinecone_index.query(vector=query_vector, top_k=k, include_metadata=True)
    if not query_results.matches:
        raise HTTPException(status_code=404, detail="⚠️ No relevant chunks found.")

    
    context = [match.metadata.get("text", "") for match in query_results.matches]
    

    
    prompt = f"""You are an AI assistant. Use the following context to answer the user's query:

Context:
{context}

User: {query}
AI:"""

    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return {"answer": answer}
    except Exception as e:
        print(f"❌ Gemini response failed: {e}")
        raise HTTPException(status_code=500, detail="LLM response failed")
