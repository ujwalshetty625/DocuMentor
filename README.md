# DocuMentor – PDF Q&A Assistant

DocuMentor is a Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.

The system retrieves the most relevant document chunks using FAISS and generates grounded answers using Groq’s `llama-3.1-8b-instant` model.

---

## Live Demo

https://huggingface.co/spaces/UjwalShetty/DocuMentor

---

## Features

- Upload any PDF document  
- Build a semantic search index using FAISS  
- Ask natural language questions about the document  
- Answers are generated strictly from retrieved context  
- Source pages are shown alongside responses  

---

## Tech Stack

- Python  
- Gradio  
- LangChain  
- FAISS Vector Store  
- HuggingFace Sentence Transformers (MiniLM)  
- Groq LLM (`llama-3.1-8b-instant`)  

---

## How It Works

1. User uploads a PDF file  
2. PDF is split into overlapping text chunks  
3. Chunks are embedded using `all-MiniLM-L6-v2`  
4. FAISS builds a vector index for retrieval  
5. User asks a question  
6. Top relevant chunks are retrieved  
7. Groq LLM generates an answer using only retrieved context  

---

## Project Structure

```bash
docu-mentor/
│── app.py
│── requirements.txt
│── .env
│── README.md
