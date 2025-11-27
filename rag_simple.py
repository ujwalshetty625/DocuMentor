from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load environment variables (GROQ_API_KEY)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment!")


# 2. Initialize LLM (Groq)
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
)


# 3. Load the text file
with open("notes.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"[INFO] Loaded {len(raw_text)} characters from notes.txt")


# 4. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
)
chunks = text_splitter.split_text(raw_text)
print(f"[INFO] Split into {len(chunks)} chunks")


# 5. Create embeddings and build FAISS vector store
print("[INFO] Creating embeddings (this may take a bit the first time)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = FAISS.from_texts(chunks, embeddings)
print("[INFO] Vector store ready.")


# 6. Function to answer a query using RAG
def answer_query(query: str) -> str:
    # 6.1 Retrieve top-k similar chunks
    docs = vectordb.similarity_search(query, k=3)

    # 6.2 Build context string
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # 6.3 Build prompt for the LLM
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


# 7. Simple CLI loop
if __name__ == "__main__":
    print("\n✅ Simple RAG over notes.txt is ready.")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        user_q = input("You: ")
        if user_q.lower() in ["exit", "quit", "q"]:
            print("Bye! 👋")
            break

        answer = answer_query(user_q)
        print("\nAssistant:", answer)
        print("-" * 60)
