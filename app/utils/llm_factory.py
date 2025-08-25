import os
import time
from typing import Callable

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _resolve_base_url() -> str:
    # Honor explicit override first
    base = os.getenv("OPENAI_API_BASE")
    if base:
        return base.rstrip("/")
    # If LiteLLM key present, default to local proxy
    if os.getenv("LITELLM_MASTER_KEY"):
        return "http://localhost:4000/v1"
    # If OpenRouter present, default to its API
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    # Fallback to OpenAI default base (if user supplies their own key)
    return "https://api.openai.com/v1"


def _resolve_api_key() -> str:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LITELLM_MASTER_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    )


def make_llm():
    base = _resolve_base_url()
    key = _resolve_api_key()
    model = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "gpt-4o-mini"
    )
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "700"))
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))
    return ChatOpenAI(
        base_url=base,
        api_key=key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def make_emb():
    # Prefer local embeddings if requested
    if os.getenv("USE_LOCAL_EMB", "0") == "1":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model=os.getenv("LOCAL_EMB_MODEL", "nomic-embed-text"))
    else:
        base = _resolve_base_url()
        key = _resolve_api_key()
        model = os.getenv("OPENAI_EMBEDDINGS", "text-embedding-3-large")
        return OpenAIEmbeddings(base_url=base, api_key=key, model=model)


def with_retry(call: Callable, attempts: int = 3, backoff: float = 0.8):
    for i in range(attempts):
        try:
            return call()
        except Exception:
            time.sleep(backoff * (2 ** i))
    # Final fallback: local LLM if available
    try:
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"))
    except Exception as e:
        raise e
