import os
import re
import ftfy
import spacy
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document

# Load the spaCy model
# Note: The paper specifies disabling the tagger and parser during inference 
# to improve runtime efficiency (a 3x speedup with no F1 loss).
try:
    # Replace 'en_legal_ner_trf' with your actual trained model name if different
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
except Exception as e:
    print(f"Model load error: {e}. Falling back to blank model for testing.")
    nlp = spacy.blank("en")

def extract_pdf_native(file_path: str) -> str:
    """Extracts exact character coordinates from native PDFs."""
    text_content = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_content.append(extracted)
    return "\n".join(text_content)

def extract_image_ocr(file_path: str) -> str:
    """Extracts text from rasterized inputs using Tesseract OCR."""
    # The paper explicitly constrains OCR configuration to -psm 6 
    # to avoid layout hallucinations by treating the page as a single block.
    custom_config = r'--psm 6'
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def extract_docx(file_path: str) -> str:
    """Parses DOCX/XML formats."""
    doc = Document(file_path)
    text_content = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(text_content)

def normalize_text(text: str) -> str:
    """
    Implements Pre-inference Pruning to resolve noise and encoding issues.
    """
    # 1. Regex pruning: Strip recurring footer patterns and watermarks
    # (Adjust the regex based on specific contract templates you process)
    text = re.sub(r'(?i)page \d+ of \d+', '', text)
    
    # 2. Unicode resolution: Repair broken Unicode sequences (e.g., Windows-1252)
    text = ftfy.fix_text(text)
    
    # 3. Whitespace collapse: Shrink excessive spacing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def reconstruct_clauses(doc) -> list:
    """
    Implements Algorithm 2: Reconstruction of Clause from Boundary Tokens.
    Executes a linear scan O(N) over predicted offsets.
    """
    clauses = []
    # If the model uses a specific entity label for clause starts (e.g., "CLAUSE_START")
    # For fallback testing, we use sentence boundaries or a generic entity approach.
    
    # Assuming 'CLAUSE_START' is the custom NER label trained in the spaCy model
    start_idx = 0
    for ent in doc.ents:
        if ent.label_ == "CLAUSE_START":
            if ent.start_char > start_idx:
                # Splice the raw text string from the detected indices
                clause_text = doc.text[start_idx:ent.start_char].strip()
                if clause_text:
                    clauses.append({"id": len(clauses) + 1, "text": clause_text})
            start_idx = ent.start_char
            
    # Append the final segment
    if start_idx < len(doc.text):
        final_clause = doc.text[start_idx:].strip()
        if final_clause:
             clauses.append({"id": len(clauses) + 1, "text": final_clause})
             
    # Fallback if no specific tags are found (for initial testing without the custom model)
    if not clauses and doc.text:
         clauses.append({"id": 1, "text": doc.text})

    return clauses

def extract_clauses(file_path: str, ext: str) -> list:
    """
    Main pipeline: Routing -> Extraction -> Normalization -> Inference -> Reconstruction
    """
    raw_text = ""
    
    # 1. Format-Specific Routing
    if ext == "pdf":
        raw_text = extract_pdf_native(file_path)
    elif ext in ["jpg", "jpeg", "png"]:
        raw_text = extract_image_ocr(file_path)
    elif ext == "docx":
        raw_text = extract_docx(file_path)
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # 2. Pre-inference Pruning
    clean_text = normalize_text(raw_text)

    # 3. Segmentation Engine Constraints
    # Inference using only the NER component
    doc = nlp(clean_text)

    # 4. JSON Reconstruction
    structured_clauses = reconstruct_clauses(doc)

    return structured_clauses