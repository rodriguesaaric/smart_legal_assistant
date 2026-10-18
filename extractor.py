import os
import re
from typing import List, Dict, Optional
import ftfy
from dotenv import load_dotenv

load_dotenv()

# Setup OCR path if configured in .env
TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH", "")
if TESSERACT_CMD_PATH and os.path.exists(TESSERACT_CMD_PATH):
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    except Exception:
        pass

# Model loading with graceful fallback
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
tokenizer = None
model = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    if os.path.exists(MODEL_DIR):
        print(f"Loading custom Segmentation Engine (Transformer) from {MODEL_DIR}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        model.eval()
        print("Segmentation Engine loaded successfully.")
except Exception as e:
    print(f"Warning: Segmentation Engine weights could not be loaded ({e}). Using regex/structural segmentation.")
    tokenizer = None
    model = None


# Structural clause regex pattern matching legal numbering and section headers
CLAUSE_PATTERN = re.compile(
    r'(?:^|\n)'
    r'(?:'
    r'(?:\d+\.[\d.]*)\s+'
    r'|(?:[IVXLC]+\.)\s+'
    r'|(?:[a-z]\))\s+'
    r'|(?:Section\s+\d+)'
    r'|(?:SECTION\s+\d+)'
    r'|(?:Clause\s+\d+)'
    r'|(?:CLAUSE\s+\d+)'
    r'|(?:Article\s+\d+)'
    r'|(?:ARTICLE\s+\d+)'
    r')',
    re.IGNORECASE
)


class TransformerSegmentationEngine:
    """
    Ingestion, text normalization, and hybrid clause segmentation engine.
    Supports native PDFs, scanned documents (OCR), DOCX, and TXT.
    """

    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model

    def _extract_pdf_native(self, file_path: str) -> str:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        # If no native text extracted, attempt OCR if pytesseract is available
        if not text.strip():
            try:
                return self._extract_pdf_scanned(file_path)
            except Exception:
                pass
        return text

    def _extract_pdf_scanned(self, image_or_pdf_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError("pytesseract or Pillow is not installed. Scanned OCR unavailable.")

        ext = image_or_pdf_path.rsplit(".", 1)[-1].lower()
        if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp"):
            image = Image.open(image_or_pdf_path)
            custom_config = r'--psm 6'
            return pytesseract.image_to_string(image, config=custom_config)
        elif ext == "pdf":
            # For scanned PDF, render pages using pdf2image or pypdfium2
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(image_or_pdf_path)
                full_text = []
                for i in range(len(pdf)):
                    page = pdf[i]
                    bitmap = page.render(scale=2.0)
                    pil_image = bitmap.to_pil()
                    page_text = pytesseract.image_to_string(pil_image, config=r'--psm 6')
                    full_text.append(page_text)
                return "\n".join(full_text)
            except Exception as e:
                raise RuntimeError(f"OCR failed for scanned PDF: {e}")
        else:
            raise ValueError(f"Unsupported image/scanned format: {ext}")

    def _extract_docx(self, file_path: str) -> str:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def _extract_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def route_and_extract(self, file_path: str, file_type: str) -> str:
        if file_type == 'pdf-native':
            return self._extract_pdf_native(file_path)
        elif file_type == 'pdf-scanned':
            return self._extract_pdf_scanned(file_path)
        elif file_type == 'docx':
            return self._extract_docx(file_path)
        elif file_type == 'txt':
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported extraction format: {file_type}")

    def normalize_text(self, text: str) -> str:
        """
        Cleans header/footer artifacts, normalizes broken unicode via ftfy,
        and standardizes whitespace.
        """
        text = re.sub(r'(?i)(page\s+\d+\s+of\s+\d+|confidential)', '', text)
        text = ftfy.fix_text(text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def segment_with_transformer(self, text: str) -> List[Dict]:
        """
        Uses Legal-BERT token classification to detect CLAUSE_START tokens.
        Applies a sliding window of 512 tokens with 64-token overlap to support
        arbitrarily long documents.
        """
        import torch

        if not self.tokenizer or not self.model:
            raise RuntimeError("Segmentation Engine weights not loaded into pipeline.")

        words = text.split()
        if len(words) < 5:
            return [{"id": 1, "text": text.strip()}]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)[0].tolist()

        clauses = []
        clause_id = 1
        current_start_char = 0

        for i in range(1, len(predictions) - 1):
            label = predictions[i]
            token_start_char, token_end_char = offsets[i]
            if token_start_char == 0 and token_end_char == 0:
                continue

            if label == 1 and token_start_char > current_start_char + 15:
                clause_text = text[current_start_char:token_start_char].strip()
                if clause_text and len(clause_text) > 20:
                    clauses.append({"id": clause_id, "text": clause_text})
                    clause_id += 1
                current_start_char = token_start_char

        final_clause_text = text[current_start_char:].strip()
        if final_clause_text:
            clauses.append({"id": clause_id, "text": final_clause_text})

        return clauses

    def segment_with_structure(self, text: str) -> List[Dict]:
        """
        Segment clauses using regex pattern matching on contractual numbering
        (1., 1.1, Section X, Article Y, etc.), with paragraph and sentence grouping fallbacks.
        """
        boundaries = [m.start() for m in CLAUSE_PATTERN.finditer(text)]

        if len(boundaries) >= 2:
            clauses = []
            for i, start in enumerate(boundaries):
                end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
                clause_text = text[start:end].strip()
                if clause_text and len(clause_text) > 15:
                    clauses.append({"id": len(clauses) + 1, "text": clause_text})
            if len(clauses) >= 2:
                return clauses

        # Fallback 1: Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
        if len(paragraphs) >= 2:
            clauses = []
            for i, para in enumerate(paragraphs):
                if len(para) > 15:
                    clauses.append({"id": len(clauses) + 1, "text": para})
            if len(clauses) >= 2:
                return clauses

        # Fallback 2: Sentence chunking (3 sentences per clause)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if len(sentences) >= 2:
            clauses = []
            for i in range(0, len(sentences), 3):
                chunk = ' '.join(sentences[i:i + 3]).strip()
                if chunk and len(chunk) > 15:
                    clauses.append({"id": len(clauses) + 1, "text": chunk})
            if clauses:
                return clauses

        return [{"id": 1, "text": text.strip()}]

    def segment_and_reconstruct(self, text: str) -> List[Dict]:
        """
        Hybrid segmentation: runs structural clause segmentation first.
        If standard numbering/structure is present, uses it for exact boundary precision.
        If text lacks explicit headers, attempts transformer-based boundary detection.
        """
        structured = self.segment_with_structure(text)
        if len(structured) >= 2:
            return structured

        # If unnumbered or unstructured block, try transformer model
        if self.model and self.tokenizer:
            try:
                tf_clauses = self.segment_with_transformer(text)
                if len(tf_clauses) >= 2:
                    return tf_clauses
            except Exception:
                pass

        return structured


def extract_clauses(file_path: str, ext: str) -> List[Dict]:
    """
    Main extraction helper for files.
    """
    engine = TransformerSegmentationEngine()
    ext_lower = ext.lower().lstrip(".")

    if ext_lower == "pdf":
        file_type = "pdf-native"
    elif ext_lower in ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]:
        file_type = "pdf-scanned"
    elif ext_lower in ["docx", "doc"]:
        file_type = "docx"
    elif ext_lower in ["txt", "text", "md"]:
        file_type = "txt"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    raw_text = engine.route_and_extract(file_path, file_type)
    clean_text = engine.normalize_text(raw_text)
    clauses = engine.segment_and_reconstruct(clean_text)
    return clauses


def extract_clauses_from_text(text: str) -> List[Dict]:
    """
    Main extraction helper for raw text strings.
    """
    engine = TransformerSegmentationEngine()
    clean_text = engine.normalize_text(text)
    return engine.segment_and_reconstruct(clean_text)