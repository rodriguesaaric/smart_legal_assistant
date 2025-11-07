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


# --- New Import for AI Model ---
import spacy
load_dotenv()

# ----------- File readers -----------
TESSERACT_PATH = os.getenv("TESSERACT_CMD_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
print(f"DEBUG: Using Tesseract Path: {TESSERACT_PATH}")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# It will load from the .env file thanks to load_dotenv()

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
MODEL_PATH = "./training/clause-model"
NLP = None
try:
    NLP = spacy.load(MODEL_PATH)
    print(f"DEBUG: Successfully loaded local AI model from {MODEL_PATH}")
except IOError:
    print(f"ERROR: Could not load model from {MODEL_PATH}.")
    print("Did you run the 'train.py' script first?")
except ImportError:
    print("ERROR: spaCy not found. Please run 'pip install spacy'")


# ---------------------------------------------------------
# --- 🤖 AI Clause segmentation (REPLACED Regex) ---
# ---------------------------------------------------------
def split_into_clauses(text: str) -> List[Dict]:
    """
    Splits a document's text into clauses using our custom-trained
    local spaCy model.
    """
    if not NLP:
        print("ERROR: AI Model (NLP) is not loaded. Cannot split clauses.")
        return [{"clause_id": "ERROR: Model not loaded", "text": text}]
        
    if not text or text.isspace():
        return []

    print("DEBUG: Starting AI clause segmentation...")
    doc = NLP(text)
    
    # Get the *start positions* of all detected clause headings
    clause_starts = [ent.start_char for ent in doc.ents if ent.label_ == "CLAUSE_START"]
    clause_starts.sort()

    clauses = []
    
    if not clause_starts:
        # No clauses found, return all text as one blob
        return [{"clause_id": "Unknown", "text": text.strip()}]

    # 1. Handle Preamble (text before the first clause)
    if clause_starts[0] > 0:
        preamble_text = text[:clause_starts[0]].strip()
        if preamble_text:
            clauses.append({"clause_id": "Preamble", "text": preamble_text})

    # 2. Loop through all found clauses
    for i in range(len(clause_starts)):
        start_char = clause_starts[i]
        
        # Determine the end of the clause
        if i + 1 < len(clause_starts):
            # End is the start of the next clause
            end_char = clause_starts[i+1]
        else:
            # This is the last clause, so it goes to the end of the text
            end_char = len(text)
            
        # Get the full text of this clause
        clause_text_full = text[start_char:end_char].strip()
        
        if not clause_text_full:
            continue

        # Try to find the clause_id (the first line)
        first_line = clause_text_full.split('\n', 1)[0].strip()
        
        clauses.append({"clause_id": first_line, "text": clause_text_full})

    print(f"DEBUG: Found {len(clauses)} clauses.")
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
