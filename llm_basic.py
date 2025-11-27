from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# 1. Load .env (GROQ_API_KEY)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment!")

print("Using GROQ_API_KEY starting with:", api_key[:8], "...")


# 2. Create LLM client with a CURRENTLY supported model
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # updated model name
    temperature=0.2,
)

# 3. Send a simple test message
messages = [
    HumanMessage(
        content="Explain what is deep learning in 2-3 simple sentences, like I'm a college student."
    )
]

try:
    response = llm.invoke(messages)
    print("\nAssistant:", response.content)
except Exception as e:
    print("\nERROR FROM GROQ / LANGCHAIN:")
    print(repr(e))
