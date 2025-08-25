import os, hmac, logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status, Depends
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- load local env file (git-ignored) ---
load_dotenv(".env.local")  # safe in dev; remove for prod if you prefer

# --- fetch the key from env ---
API_KEY = os.getenv("BRANDID", "")  # matches your Developer Portal secret name

def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

async def verify_key(request: Request):
    # Expect "Authorization: Bearer <API_KEY>"
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {API_KEY}"
    if not API_KEY or not _eq(auth, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

app = FastAPI(title="Brand Identity API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

# ---- your schemas ----
class SearchRequest(BaseModel):
    query: str
    k: int = 5

class AnswerRequest(BaseModel):
    question: str
    k: int = 5

# ---- routes (protected) ----
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/search", dependencies=[Depends(verify_key)])
async def search(req: SearchRequest):
    # TODO: real search
    return {"matches": []}

@app.post("/answer", dependencies=[Depends(verify_key)])
async def answer(req: AnswerRequest):
    # TODO: real answer
    return {"text": "stub", "citations": []}
