import os
from dotenv import load_dotenv
from openai import AzureOpenAI


# Load .env.local first if present, else fall back to .env
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()


def main():
    client = AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        messages=[{"role": "user", "content": "Hello from BrandID RAG agent!"}],
    )

    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

