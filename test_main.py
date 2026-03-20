import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ---------------- BASIC ROUTES ----------------
def test_home():
    res = client.get("/")
    assert res.status_code in [200]


def test_health():
    res = client.get("/health")
    assert res.status_code in [200, 502]


def test_models():
    res = client.get("/models")
    assert res.status_code in [200, 502]


# ---------------- CHAT API ----------------
def test_chat_valid_chat_mode():
    res = client.post("/chat", json={
        "prompt": "Hello",
        "mode": "chat"
    })
    assert res.status_code in [200, 502]


def test_chat_valid_generate_mode():
    res = client.post("/chat", json={
        "prompt": "Hello",
        "mode": "generate"
    })
    assert res.status_code in [200, 502]


def test_chat_missing_prompt():
    res = client.post("/chat", json={})
    assert res.status_code == 422


def test_chat_default_model():
    res = client.post("/chat", json={
        "prompt": "Test default model"
    })
    assert res.status_code in [200, 502]


def test_chat_invalid_model():
    res = client.post("/chat", json={
        "prompt": "Hello",
        "model": "fake-model"
    })
    assert res.status_code in [200, 502]


# ---------------- EDGE CASES ----------------
def test_large_prompt():
    big_text = "Hello " * 1000
    res = client.post("/chat", json={
        "prompt": big_text
    })
    assert res.status_code in [200, 502]


def test_invalid_json(): 
    res = client.post("/chat", content="notjson")
    assert res.status_code == 422
