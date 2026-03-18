# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import uvicorn

# Configuration
LOCAL_MODEL_API = os.environ.get("LOCAL_MODEL_API", "http://localhost:11434") 
# Ollama default HTTP: 11434. Change if you use another runner.
OLLAMA_GENERATE_PATH = "/api/generate"

app = FastAPI(title="LAN AI Server")

# Allow LAN access from other devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict if you want e.g. ["http://192.168.29.*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    # Serve frontend file
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(path)


@app.post("/chat")
async def chat_endpoint(req: Request):
    """
    Proxy /chat to the local model runner HTTP API.
    Expects JSON: {"prompt": "<text>", "max_tokens": 512, "temperature": 0.2}
    Returns JSON with: {"answer": "..."}
    """
    data = await req.json()
    prompt = data.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    # Basic generation payload; adjust according to your runner's API
    payload = {
        "model": data.get("model", "mistral"),  # name used by your runner
        "prompt": prompt,
        "max_tokens": data.get("max_tokens", 512),
        "temperature": data.get("temperature", 0.2),
        # add other runner params if needed
    }

    try:
        resp = requests.post(
            LOCAL_MODEL_API + OLLAMA_GENERATE_PATH,
            json=payload,
            timeout=120  # allow slow CPU inference
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Model runner unreachable: {e}")

    if resp.status_code != 200:
        # forward error
        raise HTTPException(status_code=502, detail=f"Runner error: {resp.status_code} {resp.text}")

    # Try to extract text from response (Ollama-style or generic)
    j = resp.json()
    # Ollama returns a structure which may vary; try common fields:
    answer = None
    if isinstance(j, dict):
        # Ollama often returns {"id":..., "model":..., "choices":[{"id":...,"content":[{"type":"output_text","text":"..."}]}]}
        choices = j.get("choices") or []
        if choices:
            # try to pull text chunks
            out = []
            for c in choices:
                # content may be list of dicts
                content = c.get("content") or []
                for piece in content:
                    if piece.get("type") in ("output_text", "message") and piece.get("text"):
                        out.append(piece.get("text"))
                    elif isinstance(piece, str):
                        out.append(piece)
            if out:
                answer = "".join(out)
        # fallback simple fields
        if not answer:
            answer = j.get("text") or j.get("output") or j.get("response")
    else:
        # not a dict, try raw string
        answer = str(j)

    if answer is None:
        # last fallback: stringify whole response
        answer = resp.text

    return JSONResponse({"answer": answer})


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Bind to 0.0.0.0 so LAN devices can connect
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)