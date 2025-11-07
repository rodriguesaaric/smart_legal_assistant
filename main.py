from fastapi import FastAPI, UploadFile, File
import shutil
import os
from extractor import extract_clauses
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="Clause Extractor Service")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract_clauses/")
async def extract(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = file.filename.split(".")[-1].lower()
    # 2. RUN THE BLOCKING FUNCTION IN A THREAD
    # This runs your AI model without freezing the server.
    clauses = await run_in_threadpool(extract_clauses, file_path, ext)

    # 3. Clean up the uploaded file (optional, but good practice)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not remove temp file {file_path}. {e}")

    return {
        "document_name": file.filename,
        "num_clauses": len(clauses),
        "clauses": clauses,
    }
