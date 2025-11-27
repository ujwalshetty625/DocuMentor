import os
from dotenv import load_dotenv

import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# ---------- Setup ----------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment!")

# SAME LLM SETUP AS YOUR WORKING CODE
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
)

# Global vectorstore
vectordb = None


# ---------- Your original logic (unchanged) ----------

def load_pdf(pdf_file):
    """Load PDF, split into chunks, create embeddings, build FAISS index."""
    global vectordb

    if pdf_file is None:
        raise gr.Error("Please upload a PDF first.")

    loader = PyPDFLoader(pdf_file.name)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(chunks, embeddings)

    return f"Loaded {len(pages)} pages and {len(chunks)} chunks from **{os.path.basename(pdf_file.name)}**."


def ask_question(question):
    """Use RAG (retrieval + LLM) to answer a question about the loaded PDF."""
    global vectordb

    if vectordb is None:
        return "Upload a PDF and click 'Build Index' first.", ""

    question = (question or "").strip()
    if question == "":
        return "Please enter a question about the document.", ""

    docs = vectordb.similarity_search(question, k=4)

    context = ""
    sources = []
    for d in docs:
        pg = d.metadata.get("page", "N/A")
        context += f"(Page {pg}) {d.page_content}\n\n"
        sources.append(f"- Page {pg}")

    prompt = (
        "You are DocuMentor, an assistant that answers using ONLY the context provided.\n"
        "If the answer is not in the context, say: \"I don't know based on this document.\"\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}\n"
        "Answer clearly and concisely:\n"
    )

    # EXACT CALL STYLE FROM YOUR WORKING CODE
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content, "\n".join(sources) if sources else "No clear sources found."


# ---------- Upgraded DocuMentor UI (using your logic) ----------

with gr.Blocks(title="DocuMentor – PDF Q&A") as app:
    gr.Markdown(
        """
# DocuMentor

Ask focused questions about a PDF and get grounded, source-backed answers.

**How to use:**
1. Upload a PDF in the left panel.
2. Click **Build Index** to prepare the document.
3. Ask questions in the right panel.
        """
    )

    with gr.Row():
        # Left: document setup
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Document Setup")

            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
            )

            build_btn = gr.Button("Build Index", variant="primary")

            status = gr.Markdown("No document loaded yet.")

        # Right: Q&A panel
        with gr.Column(scale=2):
            gr.Markdown("### Ask DocuMentor")

            question = gr.Textbox(
                label="Question",
                placeholder="Example: Summarize the main topic of this document.",
            )

            answer = gr.Textbox(
                label="Answer",
                lines=8,
            )

            sources = gr.Markdown(
                "Sources will appear here after you ask a question.",
                label="Sources",
            )

            ask_btn = gr.Button("Send", variant="primary")

    gr.Markdown(
        """
---
_Engine: Groq `llama-3.1-8b-instant` · Embeddings: `sentence-transformers/all-MiniLM-L6-v2`_
        """
    )

    # Wire: build index
    build_btn.click(
        fn=load_pdf,
        inputs=pdf_file,
        outputs=status,
    )

    # Wire: ask question (Enter key)
    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources],
    )

    # Wire: ask question (Send button)
    ask_btn.click(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources],
    )


if __name__ == "__main__":
    app.launch()
