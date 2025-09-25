from fastapi import FastAPI, UploadFile, File
import shutil
import os
from extractor import extract_clauses

app = FastAPI(title="Clause Extractor Service")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract_clauses/")
async def extract(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = file.filename.split(".")[-1].lower()
    clauses = extract_clauses(file_path, ext)

    return {
        "document_name": file.filename,
        "num_clauses": len(clauses),
        "clauses": clauses,
    }
