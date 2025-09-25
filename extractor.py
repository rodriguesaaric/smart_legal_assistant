import re
import pdfplumber
from docx import Document
from typing import List, Dict
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
from dotenv import load_dotenv

import markdown2
from bs4 import BeautifulSoup

load_dotenv()

# ----------- File readers -----------
TESSERACT_PATH = os.getenv("TESSERACT_CMD_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
print(f"DEBUG: Using Tesseract Path: {TESSERACT_PATH}")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
def extract_text_from_pdf(file_path: str) -> str:
    text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception:
        pass

    # If no text → scanned PDF fallback
    if not text:
        images = convert_from_path(file_path,poppler_path=POPPLER_PATH)
        for img in images:
            text.append(pytesseract.image_to_string(img))

    return "\n".join(text)

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_image(file_path: str) -> str:
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)

# --- New Function to handle Markdown and HTML ---
def extract_text_from_markdown_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Convert markdown to HTML first
    if file_path.lower().endswith('.md'):
        html = markdown2.markdown(content)
    else:
        html = content
        
    # Use BeautifulSoup to strip all HTML tags and get plain text
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator='\n', strip=True)

# ----------- Clause segmentation -----------
def split_into_clauses(text: str) -> List[Dict]:
    clauses = []
    cur = []
    clause_id = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect numbered clause heading (e.g., "1.", "2.1", "Section 3")
        match = re.match(r"^(\d+(\.\d+)*|Section\s+\d+)", line, re.IGNORECASE)
        if match:
            if cur:
                clauses.append({"clause_id": clause_id, "text": " ".join(cur)})
            clause_id = match.group(0)  # first token = number
            cur = [line]
        else:
            cur.append(line)

    if cur:
        clauses.append({"clause_id": clause_id, "text": " ".join(cur)})
    return clauses

# ----------- Main function -----------
def extract_clauses(file_path: str, file_type: str) -> List[Dict]:
    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_type == "docx":
        text = extract_text_from_docx(file_path)
    elif file_type == "txt":
        text = extract_text_from_txt(file_path)
    elif file_type in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(file_path)
    elif file_type in ["md", "html"]:
        text = extract_text_from_markdown_html(file_path)
    else:
        raise ValueError("Unsupported file type")

    clauses = split_into_clauses(text)
    return clauses
