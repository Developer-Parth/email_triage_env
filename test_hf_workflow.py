#!/usr/bin/env python3
"""
Test the full workflow on HuggingFace Space to ensure the deployed environment works correctly.
"""
import requests
import json
import sys

HF_SPACE_URL = "https://developer-parth-email-triage-env.hf.space"

def test_reset():
    """Test the reset endpoint with empty POST."""
    print("Testing /reset endpoint...")
    try:
        response = requests.post(f"{HF_SPACE_URL}/reset", json={})
        response.raise_for_status()
        data = response.json()
        print(f"[OK] Reset successful (status {response.status_code})")
        print(f"  Task type: {data.get('info', {}).get('task_type', 'unknown')}")
        print(f"  Email ID: {data.get('observation', {}).get('email_id', 'unknown')}")
        return data
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}")
        if hasattr(e, 'response'):
            print(f"  Response: {e.response.text}")
        return None

def test_state():
    """Test the state endpoint."""
    print("\nTesting /state endpoint...")
    try:
        response = requests.get(f"{HF_SPACE_URL}/state")
        response.raise_for_status()
        data = response.json()
        print(f"[OK] State successful (status {response.status_code})")
        print(f"  Step count: {data.get('step_count', 'unknown')}")
        return data
    except Exception as e:
        print(f"[ERROR] State failed: {e}")
        return None

def test_step(reset_data):
    """Test taking a step with a valid action."""
    print("\nTesting /step endpoint...")
    if not reset_data:
        print("[ERROR] Cannot test step without reset data")
        return None
    
    # Create a simple action based on the task type
    task_type = reset_data.get('info', {}).get('task_type', 'easy')
    
    if task_type == 'easy':
        # Classification action for easy task
        action_data = {"action_type": "classify", "label": "work"}
    elif task_type == 'medium':
        # Priority action for medium task
        action_data = {"action_type": "prioritize", "level": "medium"}
    else:  # hard
        # Full triage action for hard task
        action_data = {"action_type": "reply", "label": "work", "level": "medium", "text": "I'll review this"}
    
    # Wrap in "action" field as required by StepRequest
    request_body = {"action": action_data}
    
    try:
        response = requests.post(f"{HF_SPACE_URL}/step", json=request_body)
        response.raise_for_status()
        data = response.json()
        print(f"[OK] Step successful (status {response.status_code})")
        print(f"  Reward: {data.get('reward', {}).get('total', 'unknown')}")
        print(f"  Done: {data.get('done', 'unknown')}")
        return data
    except Exception as e:
        print(f"[ERROR] Step failed: {e}")
        if hasattr(e, 'response'):
            print(f"  Response: {e.response.text}")
        return None

def test_health():
    """Test the health endpoint."""
    print("\nTesting /health endpoint...")
    try:
        response = requests.get(f"{HF_SPACE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"[OK] Health check successful: {data.get('status', 'unknown')}")
        return data
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return None

def test_spec():
    """Test the spec endpoint."""
    print("\nTesting /spec endpoint...")
    try:
        response = requests.get(f"{HF_SPACE_URL}/spec")
        response.raise_for_status()
        data = response.json()
        print(f"[OK] Spec successful (status {response.status_code})")
        print(f"  Environment: {data.get('name', 'unknown')} v{data.get('version', 'unknown')}")
        return data
    except Exception as e:
        print(f"[ERROR] Spec failed: {e}")
        return None

def main():
    print("=" * 60)
    print("HuggingFace Space Workflow Test")
    print("=" * 60)
    
    # Test all endpoints
    health_data = test_health()
    spec_data = test_spec()
    reset_data = test_reset()
    state_data = test_state()
    
    if reset_data:
        step_data = test_step(reset_data)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Health: {'OK' if health_data else 'ERROR'}")
    print(f"  Spec: {'OK' if spec_data else 'ERROR'}")
    print(f"  Reset: {'OK' if reset_data else 'ERROR'}")
    print(f"  State: {'OK' if state_data else 'ERROR'}")
    print(f"  Step: {'OK' if reset_data and 'step_data' in locals() and step_data else 'ERROR'}")
    
    all_passed = all([health_data, spec_data, reset_data, state_data])
    if all_passed and reset_data and 'step_data' in locals() and step_data:
        print("\n[SUCCESS] All tests passed! The HuggingFace Space is working correctly.")
        return 0
    else:
        print("\n[FAILURE] Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())