import os
import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file first
load_dotenv()

# ---------------- CONFIG ----------------
LOCAL_MODEL_API = os.getenv("LOCAL_MODEL_API", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral")
APP_API_KEY = os.getenv("APP_API_KEY", "changeme123")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

OLLAMA_CHAT = "/api/chat"
OLLAMA_GENERATE = "/api/generate"
OLLAMA_TAGS = "/api/tags"
TIMEOUT = 400

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe key log — never print the actual key
logger.info("GEMINI_API_KEY loaded: %s", "yes" if GEMINI_API_KEY else "NO - missing!")
logger.info("APP_API_KEY loaded: %s", "yes" if APP_API_KEY != "changeme123" else "WARNING - still using default key!")

# ---------------- GEMINI CLIENT ----------------
gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.warning(f"Gemini client failed to initialize: {e}")

# ---------------- AUTH ----------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != APP_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

# ---------------- FASTAPI LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(TIMEOUT, connect=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.client = client
        yield

app = FastAPI(title="LAN AI Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ---------------- MODELS ----------------
class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    mode: Literal["chat", "generate"] = "chat"
    system: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    model: str
    mode: str

# ---------------- HELPERS ----------------
async def ollama_get(path: str):
    try:
        res = await app.state.client.get(f"{LOCAL_MODEL_API}{path}")
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error(f"Ollama GET failed: {e}")
        raise HTTPException(502, f"Ollama error: {e}")

async def ollama_post(path: str, payload: dict):
    try:
        res = await app.state.client.post(f"{LOCAL_MODEL_API}{path}", json=payload)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error(f"Ollama POST failed: {e}")
        raise HTTPException(502, f"Ollama error: {e}")

def extract_text(data: Dict[str, Any]):
    if "message" in data:
        return data["message"].get("content", "")
    if "response" in data:
        return data["response"]
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0].get("content", "")
    return ""

def choose_model(prompt: str, requested_model: Optional[str]):
    if requested_model:
        return requested_model
    prompt_lower = prompt.lower()
    if len(prompt) < 20:
        return "gemma:2b"
    elif "code" in prompt_lower or "algorithm" in prompt_lower:
        return "gemini-pro"
    else:
        return "gemini-flash"

# ---------------- ROUTES ----------------
@app.get("/")
async def home():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"message": "Server running"}

@app.get("/models")
async def get_models(api_key: str = Depends(verify_api_key)):
    ollama_models = []
    gemini_models = []

    try:
        data = await ollama_get(OLLAMA_TAGS)
        ollama_models = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.warning(f"Ollama models fetch failed: {e}")

    try:
        if gemini_client:
            gemini_models = [
                m.name.replace("models/", "")
                for m in gemini_client.models.list()
                if "gemini" in m.name
            ]
    except Exception as e:
        logger.warning(f"Gemini models fetch failed: {e}")

    return {"ollama": ollama_models, "gemini": gemini_models}

@app.get("/health")
async def health():
    try:
        data = await ollama_get(OLLAMA_TAGS)
        return {
            "status": "ok",
            "models": len(data.get("models", [])),
            "default_model": DEFAULT_MODEL
        }
    except:
        return {"status": "error"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    model = choose_model(req.prompt, req.model)

    try:
        if "gemini" in model and gemini_client:
            gemini_model = "gemini-2.5-pro" if "pro" in model else "gemini-2.5-flash"
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=req.prompt
            )
            answer = response.text
            return ChatResponse(answer=answer, model=gemini_model, mode="gemini")

        if req.mode == "chat":
            payload = {
                "model": model,
                "messages": [
                    *([{"role": "system", "content": req.system}] if req.system else []),
                    {"role": "user", "content": req.prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "num_predict": req.max_tokens
                }
            }
            data = await ollama_post(OLLAMA_CHAT, payload)
        else:
            payload = {
                "model": model,
                "prompt": req.prompt,
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "num_predict": req.max_tokens
                },
	        "keep_alive": "2m"
            }
            data = await ollama_post(OLLAMA_GENERATE, payload)

        answer = extract_text(data)
        if not answer:
            raise HTTPException(500, "No response from model")

        return ChatResponse(answer=answer, model=model, mode=req.mode)

    except Exception as e:
        logger.error(f"Primary model failed: {e}")
        try:
            data = await ollama_post(OLLAMA_GENERATE, {
                "model": "mistral",
                "prompt": req.prompt,
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "num_predict": req.max_tokens
                }
            })
            answer = extract_text(data)
            return ChatResponse(answer=answer, model="fallback-mistral", mode="fallback")
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise HTTPException(500, f"All models failed: {str(e)}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
