#!/usr/bin/env python3
"""Test the /reset endpoint locally."""

import json
import subprocess
import time
import requests
import sys

def test_reset_endpoint():
    """Test the /reset endpoint with empty POST."""
    
    # Start the server in background
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "email_triage_env.server:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give server time to start
    time.sleep(3)
    
    try:
        # Test 1: Empty POST request
        print("\nTest 1: Empty POST to /reset")
        response = requests.post("http://127.0.0.1:8000/reset", json={})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"OK: Got observation and info")
            print(f"  Observation keys: {list(data.get('observation', {}).keys())}")
            print(f"  Info: {data.get('info', {})}")
            
            # Validate structure
            if "observation" not in data:
                print("ERROR: Missing 'observation' field")
                return False
            if "info" not in data:
                print("ERROR: Missing 'info' field")
                return False
            print("  Structure validation: PASS")
        else:
            print(f"ERROR: Failed with status {response.status_code}")
            return False
        
        # Test 2: POST with explicit parameters
        print("\nTest 2: POST with task='easy', seed=42")
        response = requests.post("http://127.0.0.1:8000/reset", json={"task": "easy", "seed": 42})
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("OK: Success with explicit parameters")
        
        # Test 3: GET /health
        print("\nTest 3: GET /health")
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    finally:
        # Kill server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_reset_endpoint()