
from io import BytesIO

from pypdf import PdfReader
from docx import Document

import pymupdf
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def extract_text(uploaded_file):
    """
    Extract text from a Streamlit uploaded file.
    Supports PDF, DOCX and TXT.
    """

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    
    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text(
                extraction_mode="layout"
            ) or ""

            if not text.strip():
                with pymupdf.open(
                    stream=file_bytes,
                    filetype="pdf"
                ) as pdf_document:
                    pdf_page = pdf_document.load_page(
                        page_number - 1
                    )
                    pixmap = pdf_page.get_pixmap(
                        dpi=200
                    )

                    from PIL import Image

                    image = Image.open(
                        BytesIO(pixmap.tobytes("png"))
                    )

                    text = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    )

            if text.strip():
                pages.append(
                    f"--- Page {page_number} ---\n"
                    f"{text.strip()}"
                )
            else:
                pages.append(
                    f"--- Page {page_number} ---\n"
                    "[No readable text found]"
                )

        return "\n\n".join(pages)

    elif filename.endswith(".docx"):
        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8-sig")

    else:
        raise ValueError(
            "Unsupported file type. Upload PDF, DOCX or TXT."
        )

def split_text_into_chunks(text, chunk_size=4000, overlap=300):
    """Split long text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks

import re


def find_relevant_chunks(document_text, question, max_chunks=3):
    """Find the most relevant evidence and prioritize direct answers."""

    if not document_text.strip():
        return []

    import re

    # Split the document into sentences or individual lines.
    parts = re.split(r"(?<=[.!?])\s+|\n+", document_text)

    # Remove repeated sentences while preserving their original order.
    unique_parts = []
    seen = set()

    for part in parts:
        part = part.strip()
        normalized = re.sub(r"\s+", " ", part).lower()

        if part and normalized not in seen:
            seen.add(normalized)
            unique_parts.append(part)

    question_lower = question.lower()

    # Identify the type of information the user is asking for.
    if re.search(r"\bwho\b|coordinator|manager|leader|person|name\b", question_lower):
        keywords = {"coordinator", "manager", "leader", "name", "person"}
    elif re.search(r"\bwhen\b|year|date|launched|started|founded", question_lower):
        keywords = {"year", "date", "launched", "started", "founded"}
    elif re.search(r"\bbudget\b|cost|price|amount|funding", question_lower):
        keywords = {"budget", "cost", "price", "amount", "funding"}
    elif re.search(r"\bhow many\b|components|parts|features", question_lower):
        keywords = {"components", "parts", "features", "includes"}
    else:
        keywords = set()

    # Give priority to sentences that contain question-specific clues.
    question_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", question_lower)
    )

    stop_words = {
        "the", "and", "for", "with", "what", "which", "who",
        "when", "where", "how", "does", "was", "were", "are",
        "this", "that", "from", "about", "please", "tell",
        "give", "document", "project"
    }
    question_words -= stop_words

    scored_parts = []

    for index, part in enumerate(unique_parts):
        part_lower = part.lower()
        part_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", part_lower))

        score = len(question_words.intersection(part_words))

        if keywords and any(keyword in part_lower for keyword in keywords):
            score += 5

        if score > 0:
            scored_parts.append((score, index, part))

    scored_parts.sort(key=lambda item: (-item[0], item[1]))

    if scored_parts:
        return [
            part for score, index, part in scored_parts[:max_chunks]
        ]

    return []
