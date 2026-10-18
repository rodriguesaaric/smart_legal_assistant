"""
ContractShield — Unified FastAPI Backend

Merges clause extraction (extractor.py), scoring/analysis (scoring/analyzer.py),
and NLQ semantic retrieval & grounded RAG chat into a single robust API server.
"""

import os
import asyncio
import re
import json
import shutil
import time
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

import extractor
from scoring.analyzer import analyze_clauses

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", HF_TOKEN)
GEMINI_MODEL = "gemini-flash-latest"

# ── App setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="ContractShield API",
    description="AI-Powered System for Clause Extraction, Contract Analysis, and Risk Assessment",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Singleton NLQ and Vector Store State ───────────────────────────
_nlq_ready = False
_embedder = None
_embedder_tokenizer = None
_chroma_client = None
_chroma_collection = None
_llm_ready = False
_current_contract_clauses: List[Dict[str, Any]] = []


def _init_nlq():
    """Initialize NLQ components (InLegalBERT embedder, ChromaDB, Gemini) on first use."""
    global _nlq_ready, _embedder, _embedder_tokenizer, _chroma_client, _chroma_collection, _llm_ready

    if _nlq_ready and _chroma_collection is not None:
        return

    try:
        from transformers import AutoTokenizer, AutoModel
        import google.generativeai as genai
        import chromadb
        from chromadb.config import Settings

        if _embedder is None or _embedder_tokenizer is None:
            print("Loading InLegalBERT for semantic retrieval...")
            embedder_name = "law-ai/InLegalBERT"
            _embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_name)
            _embedder = AutoModel.from_pretrained(embedder_name)
            _embedder.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embedder.to(device)

        if _chroma_client is None:
            _chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

        _chroma_collection = _chroma_client.get_or_create_collection(
            "contracts", metadata={"hnsw:space": "cosine"}
        )

        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                _llm_ready = True
            except Exception as e:
                print(f"Gemini API configuration warning: {e}")

        _nlq_ready = True
        print("NLQ module initialized successfully.")
    except Exception as e:
        print(f"NLQ initialization failed: {e}")


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using InLegalBERT (mean-pooled, L2-normalized)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        inputs = _embedder_tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = _embedder(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        masked = outputs.last_hidden_state * mask
        mean_pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vecs = mean_pooled.cpu().numpy().astype("float32")
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norm, 1e-12)


def _upload_clauses_to_chroma(clauses: List[Dict[str, Any]]):
    """Upload extracted clauses to ChromaDB collection for RAG semantic search."""
    global _chroma_client, _chroma_collection, _current_contract_clauses
    _init_nlq()

    if _chroma_collection is None:
        return

    _current_contract_clauses = clauses

    try:
        # Recreate collection to maintain fresh single-document context
        try:
            _chroma_client.delete_collection("contracts")
        except Exception:
            pass
        _chroma_collection = _chroma_client.get_or_create_collection(
            "contracts", metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(f"Warning recreating collection: {e}")

    ids, docs, metas = [], [], []
    for c in clauses:
        ctext = c.get("text", "").strip()
        if not ctext:
            continue
        cid = str(c.get("id", len(ids) + 1))
        ids.append(cid)
        docs.append(ctext)
        metas.append({"clause_id": cid})

    if not docs:
        return

    embeddings = _embed_texts(docs)
    _chroma_collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=metas,
    )
    print(f"Successfully uploaded {len(docs)} clauses to ChromaDB vector store.")


# ── Request / Response Models ───────────────────────────────────────

class AnalyzeTextRequest(BaseModel):
    contractText: str
    contractSource: str = ""
    userType: str = "consumer"
    companyName: Optional[str] = None


class ChatAskRequest(BaseModel):
    question: str
    k: int = 3
    contractText: Optional[str] = None


def _generate_document_analysis_sync(text: str, user_type: str, company_name: str = None) -> dict:
    """Generate high-level AI contract assessment (Pros/Cons or Negotiation advice)."""
    import time
    t_start = time.perf_counter()
    _init_nlq()

    if not _llm_ready:
        return {"text": "Document analysis unavailable — Gemini API Key not configured.", "time_ms": 0}

    try:
        import google.generativeai as genai
        if user_type == "consumer":
            prompt = (
                "Analyze the following contract and list the overall Pros and Cons from a Consumer's perspective "
                "in Markdown format. Be concise, actionable, and highlight critical liabilities or hidden clauses.\n\n"
                f"Contract:\n{text[:15000]}"
            )
        else:
            if company_name:
                prompt = (
                    f"Analyze the following contract strictly from the commercial perspective of {company_name} (The Company). "
                    "Defend their interests, identify unfavorable obligations or liability gaps, and recommend specific clause modifications "
                    "to make it balanced and protective. Output in Markdown format.\n\n"
                    f"Contract:\n{text[:15000]}"
                )
            else:
                prompt = (
                    "Analyze the following contract from a Company/Corporate perspective. "
                    "Provide specific negotiation recommendations on which clauses to modify for operational and legal safety. "
                    f"Output in Markdown format.\n\nContract:\n{text[:15000]}"
                )

        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        analysis_text = resp.text.strip()
    except Exception as e:
        print(f"Error generating document analysis: {e}")
        analysis_text = f"Error generating document analysis: {e}"

    t_end = time.perf_counter()
    return {"text": analysis_text, "time_ms": round((t_end - t_start) * 1000, 2)}


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ContractShield API running",
        "version": "2.0.0",
        "nlq_ready": _nlq_ready,
        "llm_ready": _llm_ready
    }


@app.post("/extract_clauses/")
async def extract_clauses_endpoint(file: UploadFile = File(...)):
    """
    Dedicated clause extraction endpoint (supporting PDF, DOCX, TXT, images/scanned inputs).
    Processes documents through the normalization and segmentation pipeline.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
        clauses = await run_in_threadpool(extractor.extract_clauses, file_path, ext)
        return {
            "filename": file.filename,
            "clause_count": len(clauses),
            "clauses": clauses
        }
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.post("/api/analyze")
async def analyze_contract_file(
    file: UploadFile = File(None),
    contractText: str = Form(None),
    contractSource: str = Form("Uploaded Document"),
    userType: str = Form("consumer"),
    companyName: str = Form(None)
):
    """
    Full contract analysis endpoint:
    - Accepts file upload (PDF, DOCX, TXT, Scanned Images) or raw text.
    - Extracts & normalizes clauses via extractor engine.
    - Computes 7 multi-criteria scores, weighted document score, and risk highlights.
    - Concurrently embeds clauses into ChromaDB vector store and generates Gemini document insights.
    """
    clauses = []
    full_text = ""

    if file and file.filename:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
            supported = ("pdf", "docx", "doc", "txt", "text", "md", "png", "jpg", "jpeg", "bmp", "tiff", "webp")
            if ext not in supported:
                raise HTTPException(400, f"Unsupported file type: .{ext}. Supported types: {', '.join(supported)}")

            clauses = await run_in_threadpool(extractor.extract_clauses, file_path, ext)
            full_text = "\n\n".join(c["text"] for c in clauses)
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
    elif contractText and contractText.strip():
        full_text = contractText.strip()
        clauses = extractor.extract_clauses_from_text(full_text)
    else:
        raise HTTPException(400, "Provide either a document file or contractText in the form.")

    if not clauses or not full_text.strip():
        raise HTTPException(400, "No readable legal text could be extracted from the input.")

    # Concurrently execute:
    # 1. 7-head scoring & heuristic analysis
    # 2. Document-level AI analysis (Gemini)
    # 3. ChromaDB vector store indexing for instant chat availability
    clause_task = run_in_threadpool(
        analyze_clauses, clauses, userType, contractSource
    )
    doc_analysis_task = run_in_threadpool(
        _generate_document_analysis_sync, full_text, userType, companyName
    )
    chroma_task = run_in_threadpool(
        _upload_clauses_to_chroma, clauses
    )

    result, doc_result, _ = await asyncio.gather(clause_task, doc_analysis_task, chroma_task)

    result["documentAnalysis"] = doc_result["text"]
    if "metrics" in result and "timing" in result["metrics"]:
        result["metrics"]["timing"]["document_analysis_ms"] = doc_result["time_ms"]

    return result


@app.post("/api/analyze-text")
async def analyze_contract_text(req: AnalyzeTextRequest):
    """Analyze contract from JSON body with raw text string."""
    if not req.contractText or not req.contractText.strip():
        raise HTTPException(400, "Contract text cannot be empty.")

    full_text = req.contractText.strip()
    clauses = extractor.extract_clauses_from_text(full_text)

    clause_task = run_in_threadpool(
        analyze_clauses, clauses, req.userType, req.contractSource
    )
    doc_analysis_task = run_in_threadpool(
        _generate_document_analysis_sync, full_text, req.userType, req.companyName
    )
    chroma_task = run_in_threadpool(
        _upload_clauses_to_chroma, clauses
    )

    result, doc_result, _ = await asyncio.gather(clause_task, doc_analysis_task, chroma_task)

    result["documentAnalysis"] = doc_result["text"]
    if "metrics" in result and "timing" in result["metrics"]:
        result["metrics"]["timing"]["document_analysis_ms"] = doc_result["time_ms"]

    return result


@app.post("/api/chat/ask")
async def chat_ask(req: ChatAskRequest):
    """
    RAG-grounded legal assistant endpoint.
    Retrieves top-k clauses via InLegalBERT + ChromaDB cosine similarity,
    and prompts Gemini to answer strictly based on retrieved clauses.
    """
    _init_nlq()

    if not _nlq_ready:
        raise HTTPException(503, "Chat service unavailable — NLQ module failed to initialize.")

    if not _llm_ready:
        raise HTTPException(503, "Chat service unavailable — GEMINI_API_KEY is not configured.")

    # If contractText is explicitly supplied, re-index immediately
    if req.contractText and req.contractText.strip():
        clauses = extractor.extract_clauses_from_text(req.contractText)
        await run_in_threadpool(_upload_clauses_to_chroma, clauses)

    count = _chroma_collection.count()
    if count == 0:
        # Fallback to in-memory cached clauses if available
        if _current_contract_clauses:
            await run_in_threadpool(_upload_clauses_to_chroma, _current_contract_clauses)
            count = _chroma_collection.count()

    if count == 0:
        raise HTTPException(400, "No contract clauses loaded. Please analyze a contract first.")

    # 1. Query Expansion via LLM to bridge lay phrasing to legal terminology
    try:
        import google.generativeai as genai
        expansion_prompt = (
            "Rewrite the following user question into a formal legal contract clause "
            "or proposition that would contain the answer. Preserve intent exactly. "
            f"Do NOT explain. Output ONLY the rewritten statement.\n\nQuestion: \"{req.question}\"\n"
        )
        model = genai.GenerativeModel(GEMINI_MODEL)
        expansion_resp = model.generate_content(expansion_prompt)
        expanded_query = expansion_resp.text.strip()
    except Exception as e:
        print(f"Query expansion failed, using original: {e}")
        expanded_query = req.question

    # 2. Vector Embedding & Cosine Similarity Lookup
    q_vec = _embed_texts([expanded_query])
    results = _chroma_collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=min(req.k, count),
        include=["documents", "distances", "metadatas"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return {
            "question": req.question,
            "answer": {"summary": "No relevant clauses found in the active contract.", "relevant_clauses": [], "highlights": []},
            "top_clauses": []
        }

    top_clauses = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]
    sims = [round(1.0 - float(d), 3) for d in distances]

    # 3. Grounded Context Construction
    context_blocks = []
    for doc, md in zip(top_clauses, metas):
        cid = md.get("clause_id", "?")
        context_blocks.append(f"[clause_id={cid}] {doc}")
    context = "\n\n---\n\n".join(context_blocks)

    # 4. Grounded Synthesis via LLM
    try:
        prompt = f"""You are a precise legal assistant.
Use ONLY the clauses provided below to answer the user's question.
If the answer cannot be determined from the clauses, state that clearly.

Return ONLY a valid JSON object matching this schema:
{{
  "summary": "<2-4 sentence concise markdown summary answering the question>",
  "relevant_clauses": [
    {{"id": <integer clause id>, "excerpt": "<short exact quotation from clause>"}}
  ],
  "highlights": ["<key term 1>", "<key term 2>"]
}}

User question: "{req.question}"

Clauses:
{context}
"""
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=({"response_mime_type": "application/json"} if hasattr(genai.types, "GenerationConfig") else {})
        )
        resp = model.generate_content(prompt)
        raw_text = resp.text.strip()

        try:
            parsed = json.loads(raw_text)
        except Exception:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {"summary": raw_text, "relevant_clauses": [], "highlights": []}

    except Exception as e:
        parsed = {
            "summary": f"Could not generate grounded answer: {str(e)}",
            "relevant_clauses": [],
            "highlights": []
        }

    return {
        "question": req.question,
        "expanded_query": expanded_query,
        "top_clauses": [
            {
                "id": metas[i].get("clause_id", str(i + 1)),
                "text": top_clauses[i],
                "similarity": sims[i],
            }
            for i in range(len(top_clauses))
        ],
        "answer": {
            "summary": parsed.get("summary", ""),
            "relevant_clauses": parsed.get("relevant_clauses", []),
            "highlights": parsed.get("highlights", []),
        },
    }


# ── Server Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
