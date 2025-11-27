from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load env + check API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment!")


# 2. Initialize LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.1,  # keep it more factual
)


# 3. Load PDF pages as Documents
PDF_PATH = "docs.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"{PDF_PATH} not found. Place a PDF with this name in the folder.")

print(f"[INFO] Loading PDF: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"[INFO] Loaded {len(docs)} pages from PDF.")


# 4. Split into chunks (on page_content)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)

chunks = text_splitter.split_documents(docs)
print(f"[INFO] Split into {len(chunks)} chunks.")


# 5. Build embeddings + FAISS vector store
print("[INFO] Creating embeddings (HuggingFace MiniLM)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = FAISS.from_documents(chunks, embeddings)
print("[INFO] Vector store ready.")


# 6. RAG query function (retrieval + generation)
def answer_query(query: str) -> str:
    # 6.1 Retrieve top-k relevant chunks
    docs = vectordb.similarity_search(query, k=4)

    # 6.2 Build context with small separators
    context_parts = []
    for d in docs:
        page = d.metadata.get("page", "N/A")
        context_parts.append(f"(Page {page}) {d.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # 6.3 Build prompt for LLM
    prompt = (
        "You are a helpful assistant that answers ONLY from the given context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content, docs


# 7. CLI chat loop
if __name__ == "__main__":
    print("\n✅ PDF RAG (CLI) is ready.")
    print(f"Now answering from: {PDF_PATH}")
    print("Ask questions, or type 'exit' to quit.\n")

    while True:
        user_q = input("You: ")
        if user_q.lower() in ["exit", "quit", "q"]:
            print("Bye!")
            break

        answer, source_docs = answer_query(user_q)

        print("\nAssistant:", answer)
        print("\n[Sources]")
        for i, d in enumerate(source_docs, start=1):
            page = d.metadata.get("page", "N/A")
            print(f"  {i}. Page {page}")
        print("-" * 60)
