from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from process_pdf import save_and_extract_text, split_text
from search import create_faiss_index, search_similar_chunks

# Initialize FastAPI app
app = FastAPI()

# CORS setup so you can call it from frontend like Postman or React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or set to your domain)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store text and chunks (temporary memory)
text = ""
stored_chunks = []

@app.post("/upload/")
async def upload_file(file: UploadFile):
    global text, stored_chunks

    # Extract text and file title
    text, title = save_and_extract_text(file)
    stored_chunks = split_text(text)

    # Create FAISS index with chunks
    create_faiss_index(stored_chunks)

    return {
        "status": "File processed and indexed",
        "title": title,
        "num_chunks": len(stored_chunks)
    }

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global text, stored_chunks

    if not text:
        return {"error": "No document uploaded yet."}

    # Rule for title extraction
    if "title of the case" in question.lower():
        lines = text.split('\n')
        for line in lines:
            if "vs" in line.lower() or "v." in line.lower():
                return {
                    "answers": [
                        {
                            "question": question,
                            "answer": line.strip(),
                            "matched_clause": line.strip()
                        }
                    ]
                }

    # Else: do semantic search
    top_chunks = search_similar_chunks(question, top_k=3)

    answers = []
    for chunk in top_chunks:
        answers.append({
            "question": question,
            "answer": chunk,
            "matched_clause": chunk
        })

    return {"answers": answers}
