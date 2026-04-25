import requests
import base64
import time
import json
import sys

BASE_URL = "http://localhost:45701/api"

# Authentication (Use the first token from .env for testing)
API_TOKEN = "185ba88fbe26b6d75efc33c919a31fb0a1f229b67e9c0fed"
HEADERS = {
    "Content-Type": "application/json",
    "x-m-token": API_TOKEN
}

def log_test(name, status, details=""):
    color = "\033[92m" if status == "PASS" else "\033[91m"
    print(f"{color}[{status}] {name}\033[0m {details}")

def test_health():
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            log_test("Health Check", "PASS", f"- GPUs: {resp.json().get('gpu_count')}")
            return True
        else:
            log_test("Health Check", "FAIL", f"Status: {resp.status_code}")
    except Exception as e:
        log_test("Health Check", "FAIL", str(e))
    return False

def test_lstm():
    try:
        payload = {"prompt": "The future of AI is", "max_tokens": 50}
        resp = requests.post(f"{BASE_URL}/text/lstm/generate", json=payload, headers=HEADERS)
        if resp.status_code == 200:
            text = resp.json().get("text", "")
            log_test("LSTM Text Gen", "PASS", f"Result: '{text[:50]}...'")
            return True
        else:
            log_test("LSTM Text Gen", "FAIL", f"Status: {resp.status_code} - {resp.text}")
    except Exception as e:
        log_test("LSTM Text Gen", "FAIL", str(e))
    return False

def test_mixtral():
    print("Testing Mixtral (this may take a while to load)...")
    try:
        payload = {"prompt": "Hello, who are you?", "max_tokens": 50}
        resp = requests.post(f"{BASE_URL}/text/generate", json=payload, headers=HEADERS, timeout=300)
        if resp.status_code == 200:
            text = resp.json().get("text", "")
            log_test("Mixtral Text Gen", "PASS", f"Result: '{text[:50]}...'")
            return True
        else:
            log_test("Mixtral Text Gen", "FAIL", f"Status: {resp.status_code} - {resp.text}")
    except Exception as e:
        log_test("Mixtral Text Gen", "FAIL", str(e))
    return False

def test_audio():
    try:
        payload = {"text": "Hello, this is a test of the audio system."}
        resp = requests.post(f"{BASE_URL}/audio/generate", json=payload, headers=HEADERS, timeout=60)
        if resp.status_code == 200:
            log_test("XTTS Audio Gen", "PASS", f"Duration: {resp.json().get('duration_seconds')}s")
            return True
        else:
            log_test("XTTS Audio Gen", "FAIL", f"Status: {resp.status_code} - {resp.text}")
    except Exception as e:
        log_test("XTTS Audio Gen", "FAIL", str(e))
    return False

if __name__ == "__main__":
    print("=== Multi-Model AI Platform Integration Test ===")
    
    if not test_health():
        print("Aborting tests as health check failed.")
        sys.exit(1)
        
    test_lstm()
    test_audio()
    # test_mixtral()
    
    print("\nTests complete.")
