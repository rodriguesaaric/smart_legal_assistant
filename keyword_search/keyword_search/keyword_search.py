# app.py
# pip install fastapi uvicorn transformers chromadb faiss-cpu torch openai python-multipart python-dotenv

import os
import json
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# ───────────────────────────────────────────────
# ENV + CONFIG
# ───────────────────────────────────────────────
load_dotenv()
HF_API_BASE = "https://router.huggingface.co/v1"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not set.")

client = OpenAI(base_url=HF_API_BASE, api_key=HF_TOKEN)

# ───────────────────────────────────────────────
# EMBEDDINGS
# ───────────────────────────────────────────────
print("Loading InLegalBERT for embeddings...")
embedder_name = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(embedder_name)
embedder = AutoModel.from_pretrained(embedder_name)
embedder.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder.to(device)

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

@torch.no_grad()
def embed_texts(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = embedder(**inputs)
    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    masked = token_embeddings * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    mean_pooled = summed / counts
    vecs = mean_pooled.cpu().numpy().astype("float32")
    return _l2_normalize(vecs)

# ───────────────────────────────────────────────
# VECTOR DB
# ───────────────────────────────────────────────
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(
    "indian_contracts",
    metadata={"hnsw:space": "cosine"}
)

# ───────────────────────────────────────────────
# FASTAPI SETUP
# ───────────────────────────────────────────────
app = FastAPI(title="ClauseAI – Legal Contract Query Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────
# STOP WORDS
# ───────────────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "is", "are", "to", "for", "of", "in", "on",
    "with", "this", "that", "what", "how", "does", "do", "can",
    "may", "shall", "will", "be", "by", "at", "or", "if"
}

def simple_keyword_extract(text, question):
    words = question.lower().split()
    keywords = [
        w.strip(".,!?()[]{}:;\"'")
        for w in words
        if w.strip(".,!?()[]{}:;\"'").lower() not in STOP_WORDS
    ]
    text_words = set(
        w.strip(".,!?()[]{}:;\"'").lower()
        for w in text.split()
    )
    return list({k for k in keywords if k in text_words})

def extract_highlights(top_clauses, question, summary):
    focus_words = set(
        w.strip(".,!?()[]{}:;\"'").lower()
        for w in (question + " " + summary).split()
        if w.strip(".,!?()[]{}:;\"'").lower() not in STOP_WORDS
    )

    highlights = []
    for clause in top_clauses:
        clause_lower = clause.lower()
        hits = [fw for fw in focus_words if fw in clause_lower]
        highlights.extend(hits)

    return sorted(set(highlights))

# ───────────────────────────────────────────────
# UPLOAD ENDPOINT
# ───────────────────────────────────────────────
@app.post("/upload")
async def upload_clauses(file: UploadFile = File(...)):
    global collection
    try:
        import time
        start_all = time.time()

        try:
            data = json.load(file.file)
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {str(e)}"})
        
        clauses = data.get("clauses", [])
        if not isinstance(clauses, list):
            return JSONResponse(status_code=400, content={"error": "Clauses must be a list/array"})

        try:
            chroma_client.delete_collection("indian_contracts")
        except Exception:
            pass

        collection = chroma_client.get_or_create_collection(
            "indian_contracts",
            metadata={"hnsw:space": "cosine"}
        )

        ids, docs, metas = [], [], []
        for idx, c in enumerate(clauses, start=1):
            ctext = c.get("text", "").strip()
            if not ctext:
                continue
            ids.append(str(idx))
            docs.append(ctext)
            metas.append({"clause_id": c.get("clause_id", str(idx))})

        if not docs:
            return JSONResponse(status_code=400, content={"error": "No clauses found."})

        embeddings = embed_texts(docs)

        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings.tolist(),
            metadatas=metas,
        )

        return {
            "status": "success",
            "clauses": len(docs),
            "message": f"Uploaded {len(docs)} full clauses."
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ───────────────────────────────────────────────
# ASK ENDPOINT
# ───────────────────────────────────────────────
@app.post("/ask")
async def ask_question(question: str = Form(...), k: int = Form(3)):
    global collection
    try:
        import time
        start_all = time.time()

        if collection is None:
            return JSONResponse(status_code=404, content={"error": "Upload first."})

        # ───────────────────────────────────────────────
        # QUERY EXPANSION
        # ───────────────────────────────────────────────
        expansion_prompt = f"""
Rewrite the following user question into a formal legal contract clause
or proposition that would contain the answer.

Preserve intent exactly.
Do NOT explain.
Output ONLY the rewritten statement.

Question: "{question}"
"""

        expansion_resp = client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=120,
            temperature=0.1,
        )

        expanded_query = expansion_resp.choices[0].message.content.strip()

        # Embed expanded query
        q = embed_texts([expanded_query])

        # Retrieve
        results = collection.query(
            query_embeddings=q.tolist(),
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

        if not results or not results.get("documents"):
            return JSONResponse(status_code=404, content={"error": "No clauses found."})

        top_clauses = results["documents"][0]
        distances = results["distances"][0]
        metas = results["metadatas"][0]

        sims = [round(1.0 - float(d), 3) for d in distances]

        context_blocks = []
        for doc, md in zip(top_clauses, metas):
            cid = md.get("clause_id")
            context_blocks.append(f"[clause_id={cid}] {doc}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
You are a legal contract analysis assistant.
Use only the clauses below.

Return ONLY valid JSON:

{{
  "summary": "<2-4 sentence markdown summary>",
  "relevant_clauses": [
    {{"id": <number>, "excerpt": "<short phrase>"}}
  ],
  "highlights": ["<keyword1>", "<keyword2>"]
}}

User question: "{question}"

Clauses:
{context}
"""

        resp = client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.25,
        )

        raw_text = resp.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = {
                "summary": "Not found in provided clauses.",
                "relevant_clauses": [],
                "highlights": simple_keyword_extract(context, question)[:3],
            }

        summary = parsed.get("summary", "")
        relevant_clauses = parsed.get("relevant_clauses", [])
        highlights = parsed.get("highlights", [])

        merged_highlights = sorted(
            set(highlights + extract_highlights(top_clauses, question, summary))
        )

        return {
            "question": question,
            "expanded_query": expanded_query,
            "top_clauses": [
                {
                    "id": metas[i].get("clause_id"),
                    "text": top_clauses[i],
                    "similarity": sims[i],
                }
                for i in range(len(top_clauses))
            ],
            "answer": {
                "summary": summary,
                "relevant_clauses": relevant_clauses,
                "highlights": merged_highlights,
            },
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "ClauseAI backend running"}