import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(".env.local")

try:
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
    )
    response = llm.invoke("Hello, world!")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
