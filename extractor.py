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
import spacy


import json
from groq import Groq

load_dotenv()

# ----------- Configuration -----------
TESSERACT_PATH = os.getenv("TESSERACT_CMD_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

print(f"DEBUG: Using Tesseract Path: {TESSERACT_PATH}")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
    except Exception as e:
        print(f"WARNING: Client failed to initialize: {e}")

# ----------- File readers -----------
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

# --- Function to handle Markdown and HTML ---
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
except ImportError:
    print("ERROR: spaCy not found.")


# ---------------------------------------------------------
# --- Hybrid Clause segmentation ---
# ---------------------------------------------------------
def split_into_clauses(text: str) -> List[Dict]:
    """
    Splits a document's text into clauses.
    
    """
    
    
    clauses = []
    if NLP and text and not text.isspace():
        # print("DEBUG: Running local spaCy segmentation...") 
        doc = NLP(text)
        clause_starts = [ent.start_char for ent in doc.ents if ent.label_ == "CLAUSE_START"]
        clause_starts.sort()

        if not clause_starts:
            clauses = [{"clause_id": "Unknown", "text": text.strip()}]
        else:
            if clause_starts[0] > 0:
                clauses.append({"clause_id": "Preamble", "text": text[:clause_starts[0]].strip()})
            for i in range(len(clause_starts)):
                start = clause_starts[i]
                end = clause_starts[i+1] if i + 1 < len(clause_starts) else len(text)
                txt = text[start:end].strip()
                if txt:
                    clauses.append({"clause_id": txt.split('\n', 1)[0].strip(), "text": txt})
    
    
    if groq_client:
        
        try:
            
            prompt = f"""
            Extract legal clauses from the text below.
            Output ONLY a JSON list of objects. Format: {{"clause_id": "Title/Number", "text": "Content"}}
            
            TEXT:
            {text[:25000]} 
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a legal extraction API. Output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            
            ai_content = completion.choices[0].message.content
            ai_data = json.loads(ai_content)
            
            
            final_clauses = []
            if isinstance(ai_data, list):
                final_clauses = ai_data
            elif isinstance(ai_data, dict):
                
                for k in ["clauses", "data", "result"]:
                    if k in ai_data:
                        final_clauses = ai_data[k]
                        break
                if not final_clauses:

                    final_clauses = [ai_data]

            print(f"DEBUG: spaCy returned {len(final_clauses)} clauses.")
            return final_clauses

        except Exception as e:
            print(f"ERROR: Falling back to local spaCy output.")
            
            pass

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