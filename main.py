#main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from service import process_uploaded_files,query_faiss_index
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str  


@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    return await process_uploaded_files(files) 


@app.post("/query/")
async def query_docs(request: QueryRequest):
    answer = await query_faiss_index(request.query)
    return {"answer": answer}