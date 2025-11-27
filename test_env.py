import os
from dotenv import load_dotenv, find_dotenv

print("CWD:", os.getcwd())
print("Env file found at:", find_dotenv())

load_dotenv(find_dotenv())

print("Groq Key:", os.getenv("GROQ_API_KEY"))
