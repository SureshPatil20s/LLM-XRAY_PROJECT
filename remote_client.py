"""
Client module for communicating with a remote Colab GPU inference server.
Allows running the LLM generation and X-Ray activations on free Colab GPU
while keeping the Streamlit UI, databases, and document RAG running locally.
"""
import requests
import json
from typing import Generator, Dict, Any, Tuple, Optional


def check_remote_gpu_health(base_url: str, timeout: float = 4.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Pings the remote Colab GPU server to check connectivity and device info.
    Returns (is_healthy, status_info_dict).
    """
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        return False, {"error": "No URL provided"}

    try:
        resp = requests.get(f"{clean_url}/health", timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"HTTP {resp.status_code}: {resp.text[:100]}"}
    except Exception as e:
        return False, {"error": str(e)}


def stream_remote_chat(
    base_url: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    top_k: int = 50,
    timeout: float = 120.0
) -> Generator[str, None, None]:
    """
    Streams generated tokens from the remote Colab GPU server in real time.
    """
    clean_url = base_url.strip().rstrip("/")
    payload = {
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
        "top_k": int(top_k)
    }

    try:
        resp = requests.post(
            f"{clean_url}/stream",
            json=payload,
            stream=True,
            timeout=timeout
        )
        if resp.status_code != 200:
            yield f"[Remote GPU Error {resp.status_code}]: {resp.text[:200]}"
            return

        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk
    except Exception as e:
        yield f"\n\n[Remote GPU Connection Error]: Failed to reach Colab server at {clean_url}: {e}"


def fetch_remote_xray(
    base_url: str,
    prompt: str,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Requests mechanistic activations (X-Ray) from remote Colab GPU.
    """
    clean_url = base_url.strip().rstrip("/")
    try:
        resp = requests.post(
            f"{clean_url}/xray",
            json={"prompt": prompt},
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"error": f"Failed to connect to Colab GPU: {e}"}

