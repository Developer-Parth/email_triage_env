"""
Simple test script to validate the Email Triage environment.
"""

import sys
sys.path.insert(0, '.')

from email_triage_env import EmailTriageEnv
from email_triage_env.models import Action, EmailCategory, PriorityLevel


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing Email Triage Environment...")
    
    # Test 1: Easy task (classification)
    print("\n1. Testing Easy Task (Classification)...")
    env = EmailTriageEnv(task_type="easy")
    obs = env.reset(seed=42)
    
    print(f"  Email Subject: {obs.subject}")
    print(f"  Email Body: {obs.body[:50]}...")
    
    # Take classification action
    action = Action(action_type="classify", label=EmailCategory.WORK)
    obs, reward, done, info = env.step(action)
    
    print(f"  Reward: {reward.total}")
    print(f"  Done: {done}")
    print(f"  Is Classified: {info['is_classified']}")
    
    # Test 2: Medium task (prioritization)
    print("\n2. Testing Medium Task (Prioritization)...")
    env = EmailTriageEnv(task_type="medium")
    obs = env.reset(seed=43)
    
    action = Action(action_type="prioritize", level=PriorityLevel.HIGH)
    obs, reward, done, info = env.step(action)
    
    print(f"  Reward: {reward.total}")
    print(f"  Done: {done}")
    print(f"  Is Prioritized: {info['is_prioritized']}")
    
    # Test 3: Hard task (full triage)
    print("\n3. Testing Hard Task (Full Triage)...")
    env = EmailTriageEnv(task_type="hard")
    obs = env.reset(seed=44)
    
    # Step 1: Classify
    action1 = Action(action_type="classify", label=EmailCategory.PERSONAL)
    obs, reward1, done1, info1 = env.step(action1)
    
    # Step 2: Prioritize
    action2 = Action(action_type="prioritize", level=PriorityLevel.MEDIUM)
    obs, reward2, done2, info2 = env.step(action2)
    
    # Step 3: Reply (optional)
    action3 = Action(action_type="reply", text="Thank you for your email.")
    obs, reward3, done3, info3 = env.step(action3)
    
    print(f"  Step 1 Reward: {reward1.total}")
    print(f"  Step 2 Reward: {reward2.total}")
    print(f"  Step 3 Reward: {reward3.total}")
    print(f"  Total Reward: {reward1.total + reward2.total + reward3.total}")
    print(f"  Final Done: {done3}")
    
    # Test 4: Invalid action
    print("\n4. Testing Invalid Action...")
    env = EmailTriageEnv(task_type="easy")
    obs = env.reset(seed=45)
    
    # Invalid: classify without label
    try:
        invalid_action = Action(action_type="classify")
        obs, reward, done, info = env.step(invalid_action)
        print(f"  Invalid action handled, reward: {reward.total}")
    except Exception as e:
        print(f"  Exception (expected): {e}")
    
    print("\nAll tests completed successfully!")
    return True


def test_models():
    """Test Pydantic models."""
    print("\nTesting Pydantic Models...")
    
    # Test Observation model
    obs = {
        "email_id": "test_001",
        "subject": "Test Email",
        "body": "This is a test email body.",
        "step_count": 0
    }
    
    from email_triage_env.models import Observation
    observation = Observation(**obs)
    print(f"  Observation created: {observation.email_id}")
    
    # Test Action model
    action = Action(action_type="classify", label=EmailCategory.SPAM)
    print(f"  Action created: {action.action_type} with label {action.label}")
    
    # Test Reward model
    from email_triage_env.models import Reward
    reward = Reward(total=0.6, classification=0.6)
    print(f"  Reward created: {reward.total}")
    
    print("  Models test passed!")
    return True


if __name__ == "__main__":
    try:
        test_models()
        test_basic_functionality()
        print("\n[SUCCESS] All tests passed! Environment is working correctly.")
    except Exception as e:
        print(f"\n[FAILURE] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)