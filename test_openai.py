import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(".env.local")

# Normalize env vars (avoid trailing spaces causing connection issues)
BASE_URL = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
MODEL = (os.getenv("OPENROUTER_MODEL") or "openrouter/auto").strip()

try:
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        timeout=60,
    )
    response = llm.invoke("Hello, world!")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}\nBase URL: {BASE_URL}\nModel: {MODEL}\nKey set: {'yes' if bool(API_KEY) else 'no'}")
