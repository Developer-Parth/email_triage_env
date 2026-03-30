#!/usr/bin/env python3
"""
Final test to verify all enhancements are working correctly.
Tests: train/test split, step cost, state traps, scoring stability, reward balance.
"""

import sys
from email_triage_env import EmailTriageEnv, Observation, Action
from email_triage_env.models import EmailCategory, PriorityLevel

def test_train_test_split():
    """Test that train/test split works correctly."""
    print("Testing train/test split...")
    
    # Create environments with different dataset modes
    env_train = EmailTriageEnv(task_type="easy", dataset_mode="train")
    env_test = EmailTriageEnv(task_type="easy", dataset_mode="test")
    
    # Reset both with same seed
    obs_train = env_train.reset(seed=42)
    obs_test = env_test.reset(seed=42)
    
    # They should have different emails (train vs test split)
    if obs_train.email_id != obs_test.email_id:
        print("  [PASS] Train and test datasets are different")
    else:
        print("  [FAIL] Train and test datasets are the same")
        return False
    
    # Check dataset stats (access dataset attribute)
    stats_train = env_train.dataset.get_stats()
    stats_test = env_test.dataset.get_stats()
    
    print(f"  Train dataset: {stats_train['train_emails']} train emails, {stats_train['test_emails']} test emails")
    print(f"  Test dataset: {stats_test['train_emails']} train emails, {stats_test['test_emails']} test emails")
    
    if stats_train['train_emails'] == 10 and stats_train['test_emails'] == 10:
        print("  [PASS] Dataset has correct split (10 train, 10 test)")
    else:
        print(f"  [FAIL] Unexpected dataset sizes: {stats_train}")
        return False
    
    return True

def test_step_cost():
    """Test that step cost is applied correctly."""
    print("\nTesting step cost...")
    
    env = EmailTriageEnv(task_type="easy")
    obs = env.reset(seed=123)
    
    # Take a step
    action = Action(action_type="classify", label=EmailCategory.WORK)
    obs, reward, done, info = env.step(action)
    
    # Check that step_cost is present and negative
    if reward.step_cost is not None:
        if abs(reward.step_cost - (-0.05)) < 0.001:
            print(f"  [PASS] Step cost applied correctly: {reward.step_cost}")
        else:
            print(f"  [FAIL] Unexpected step cost: {reward.step_cost}")
            return False
    else:
        print("  [FAIL] step_cost field is None")
        return False
    
    # Check that total reward includes step cost
    # The classification reward depends on email difficulty, so we can't hardcode it
    # Instead, verify that total = sum of all non-None components
    calculated_total = 0.0
    components = ["classification", "priority", "sequence", "invalid_penalty",
                  "completion_bonus", "step_cost", "state_traps"]
    
    for comp in components:
        value = getattr(reward, comp)
        if value is not None:
            calculated_total += value
    
    if abs(reward.total - calculated_total) < 0.001:
        print(f"  [PASS] Total reward correctly calculated: {reward.total}")
        print(f"        Components: classification={reward.classification}, step_cost={reward.step_cost}")
    else:
        print(f"  [FAIL] Total reward mismatch: {reward.total} vs calculated {calculated_total}")
        return False
    
    # Also verify step cost is applied every step
    env2 = EmailTriageEnv(task_type="hard", max_steps=10)
    obs2 = env2.reset(seed=456)
    
    step_costs = []
    for i in range(3):
        action2 = Action(action_type="classify", label=EmailCategory.WORK)
        obs2, reward2, done2, info2 = env2.step(action2)
        if reward2.step_cost is not None:
            step_costs.append(reward2.step_cost)
    
    if all(abs(cost + 0.05) < 0.001 for cost in step_costs):
        print(f"  [PASS] Step cost applied consistently across steps: {step_costs}")
    else:
        print(f"  [FAIL] Inconsistent step costs: {step_costs}")
        return False
    
    return True

def test_state_traps():
    """Test that state-based traps trigger correctly."""
    print("\nTesting state-based traps...")
    
    env = EmailTriageEnv(task_type="hard")
    obs = env.reset(seed=456)
    
    # Step 1: Correct classification
    action1 = Action(action_type="classify", label=EmailCategory.WORK)
    obs1, reward1, done1, info1 = env.step(action1)
    
    # Step 2: Change classification (should trigger trap)
    action2 = Action(action_type="classify", label=EmailCategory.PERSONAL)
    obs2, reward2, done2, info2 = env.step(action2)
    
    # Check if state_traps penalty was applied
    if reward2.state_traps is not None and reward2.state_traps < 0:
        print(f"  [PASS] State trap triggered: {reward2.state_traps}")
        
        # Check the trap details in info
        if "trap_details" in info2:
            print(f"  [PASS] Trap details in info: {info2['trap_details']}")
        else:
            print("  [WARNING] No trap details in info")
    else:
        print(f"  [FAIL] State trap not triggered: state_traps={reward2.state_traps}")
        return False
    
    return True

def test_scoring_stability():
    """Test that scoring is stable with rounding."""
    print("\nTesting scoring stability...")
    
    env = EmailTriageEnv(task_type="medium")
    obs = env.reset(seed=789)
    
    # Take multiple steps and check rewards are properly rounded
    rewards = []
    for i in range(3):
        action = Action(action_type="prioritize", level=PriorityLevel.MEDIUM)
        obs, reward, done, info = env.step(action)
        rewards.append(reward.total)
    
    # Check that rewards don't have floating point weirdness
    for i, r in enumerate(rewards):
        # Convert to string and check decimal places
        r_str = f"{r:.10f}"
        # Should not have many trailing 9s or weird patterns
        if "999999" in r_str or "000001" in r_str:
            print(f"  [FAIL] Floating point instability in reward {i}: {r_str}")
            return False
    
    print(f"  [PASS] Rewards are stable: {rewards}")
    
    # Check that rounding is applied in reward components
    action = Action(action_type="prioritize", level=PriorityLevel.HIGH)
    obs, reward, done, info = env.step(action)
    
    # All reward components should be rounded to 6 decimal places
    for field in ["classification", "priority", "sequence", "step_cost", "state_traps", "completion_bonus"]:
        value = getattr(reward, field)
        if value is not None:
            # Check if value appears rounded (no long decimal tails)
            value_str = f"{value:.10f}"
            # After 6 decimal places, should be zeros or rounding
            # Simple check: convert to string with 8 decimals and compare
            rounded = round(value, 6)
            if abs(value - rounded) > 1e-9:
                print(f"  [FAIL] Field {field} not properly rounded: {value} vs {rounded}")
                return False
    
    print("  [PASS] All reward components properly rounded")
    return True

def test_reward_balance():
    """Test that reward balance is reasonable (smart > 0, bad < 0)."""
    print("\nTesting reward balance...")
    
    # Test smart agent (correct actions)
    env_smart = EmailTriageEnv(task_type="hard")
    obs = env_smart.reset(seed=111)
    
    # Smart sequence: classify correctly, prioritize correctly, reply
    actions_smart = [
        Action(action_type="classify", label=EmailCategory.WORK),
        Action(action_type="prioritize", level=PriorityLevel.MEDIUM),
        Action(action_type="reply", text="I'll handle this.")
    ]
    
    total_smart = 0.0
    for action in actions_smart:
        obs, reward, done, info = env_smart.step(action)
        total_smart += reward.total
    
    print(f"  Smart agent total reward: {total_smart:.3f}")
    
    # Test bad agent (wrong actions)
    env_bad = EmailTriageEnv(task_type="hard")
    obs = env_bad.reset(seed=111)  # Same seed, same email
    
    # Bad sequence: wrong classification, wrong priority, no reply
    actions_bad = [
        Action(action_type="classify", label=EmailCategory.SPAM),  # Wrong
        Action(action_type="prioritize", level=PriorityLevel.LOW),  # Wrong
        Action(action_type="classify", label=EmailCategory.PERSONAL),  # Change (trap)
    ]
    
    total_bad = 0.0
    for action in actions_bad:
        obs, reward, done, info = env_bad.step(action)
        total_bad += reward.total
    
    print(f"  Bad agent total reward: {total_bad:.3f}")
    
    # Check balance: smart should be positive, bad should be negative
    if total_smart > 0:
        print(f"  [PASS] Smart agent gets positive reward: {total_smart:.3f}")
    else:
        print(f"  [FAIL] Smart agent reward not positive: {total_smart:.3f}")
        return False
    
    if total_bad < 0:
        print(f"  [PASS] Bad agent gets negative reward: {total_bad:.3f}")
    else:
        print(f"  [FAIL] Bad agent reward not negative: {total_bad:.3f}")
        return False
    
    # Check that smart > bad (obviously)
    if total_smart > total_bad:
        print(f"  [PASS] Smart agent outperforms bad agent ({total_smart:.3f} > {total_bad:.3f})")
    else:
        print(f"  [FAIL] Smart agent doesn't outperform bad agent")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Final Enhancement Tests")
    print("=" * 60)
    
    tests = [
        ("Train/Test Split", test_train_test_split),
        ("Step Cost", test_step_cost),
        ("State Traps", test_state_traps),
        ("Scoring Stability", test_scoring_stability),
        ("Reward Balance", test_reward_balance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"[PASS] {test_name}")
                passed += 1
            else:
                print(f"[FAIL] {test_name}")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All enhancements working correctly!")
        return 0
    else:
        print(f"{failed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())