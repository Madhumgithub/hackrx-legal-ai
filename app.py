import streamlit as st
from process_pdf import extract_text_from_pdf
from process_docx import extract_text_from_docx
from search import query_documents  # Your FAISS + sentence-transformers logic

st.set_page_config(page_title="Legal Doc Q&A", layout="wide")

st.title("📄 AI Document Q&A System")

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
user_query = st.text_input("Ask a question about the document")

if uploaded_file and user_query:
    # Save uploaded file locally
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file.name)
    elif uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file.name)
    else:
        st.error("Unsupported file type.")
        text = ""

    if text:
        result = query_documents(text, user_query)
        st.success("Answer:")
        st.write(result)
