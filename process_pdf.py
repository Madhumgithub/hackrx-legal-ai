import os
import fitz  # PyMuPDF
from fastapi import UploadFile
from uuid import uuid4
from typing import List, Tuple

def save_and_extract_text(file: UploadFile) -> Tuple[str, str]:
    file_ext = file.filename.split(".")[-1].lower()
    file_path = f"documents/{uuid4()}.{file_ext}"

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    text = ""
    if file_ext == "pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    elif file_ext == "docx":
        import docx
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    title = extract_title(text)
    return text, title

def extract_title(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines[:10]:  # Check first 10 lines
        if "vs" in line.lower() or "v." in line.lower():
            return line.strip()
    return "Title not found"

def split_text(text: str, max_chunk_size: int = 500) -> List[str]:
    sentences = text.split(". ")
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks
