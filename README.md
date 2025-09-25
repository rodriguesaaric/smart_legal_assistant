# Clause Extractor Service

This is a FastAPI service that extracts numbered clauses from documents (.pdf, .docx, .txt, .jpg, etc.).

## Prerequisites

You must have the following software installed on your system before setting up the project.

1.  **Python 3.8+**
2.  **Tesseract-OCR**: This is required for reading text from images and scanned PDFs.
    * [Installation Guide for Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    * **Important**: Make sure to add the Tesseract installation directory to your system's PATH.

3.  **Poppler**: This is required for converting PDFs to images.
    * [Download Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
    * **Important**: You will need the path to the `bin` directory within the Poppler folder.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your environment:**
    * Create a file named `.env` in the root of the project.
    * Copy the contents of `.env.example` into it and fill in the correct paths for your system.

## Running the Service

Once set up, run the FastAPI server using uvicorn:
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

## How to Use

Send a `POST` request with a file to the `/extract_clauses/` endpoint. You can use the automatic docs at `http://127.0.0.1:8000/docs`.