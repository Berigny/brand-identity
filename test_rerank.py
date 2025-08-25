import os
from dotenv import load_dotenv
from app.utils.rerank import rerank_texts


def main():
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")
    else:
        load_dotenv()

    query = "brand color emotional token"
    texts = [
        "Our primary palette is calm blue with coherence to core nodes.",
        "Emotional tokens like #ff6699 drive novelty and attention.",
        "Mission statement and values should be clear and concise.",
        "Tertiary colors are vibrant and mutable for trends.",
    ]
    out = rerank_texts(query, texts, top_k=int(os.getenv("RERANK_TOP_K", 3)), max_tokens=int(os.getenv("CONTEXT_MAX_TOKENS", 900)))
    print("\n---\n".join(out))


if __name__ == "__main__":
    main()

