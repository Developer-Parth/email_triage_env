#!/usr/bin/env python3
"""
Test script to analyze reward balance in the Email Triage environment.
Checks if:
1. Bad agents get negative reward
2. Average agents get near 0 reward  
3. Smart agents get clearly positive reward
"""

import sys
import random
from typing import List, Tuple, Dict

sys.path.insert(0, '.')

from email_triage_env.models import Action, EmailCategory, PriorityLevel
from email_triage_env.environment import EmailTriageEnv


def test_bad_agent(task_type: str = "easy", num_episodes: int = 50) -> float:
    """Test a bad agent that always chooses wrong actions."""
    env = EmailTriageEnv(task_type=task_type, max_steps=8, dataset_mode="train")
    total_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset(seed=episode)
        episode_reward = 0.0
        done = False
        
        while not done:
            # Always choose wrong category (cycle through wrong options)
            current_state = env.state()
            if current_state:
                wrong_categories = [c for c in EmailCategory if c != current_state.ground_truth_category]
                wrong_priorities = [p for p in PriorityLevel if p != current_state.ground_truth_priority]
                
                # Alternate between wrong actions
                if random.random() < 0.5:
                    action = Action(
                        action_type="classify",
                        label=random.choice(wrong_categories) if wrong_categories else EmailCategory.WORK
                    )
                else:
                    action = Action(
                        action_type="prioritize",
                        level=random.choice(wrong_priorities) if wrong_priorities else PriorityLevel.LOW
                    )
            else:
                action = Action(action_type="classify", label=EmailCategory.SPAM)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward.total
        
        total_rewards.append(episode_reward)
    
    return sum(total_rewards) / len(total_rewards)


def test_average_agent(task_type: str = "easy", num_episodes: int = 50) -> float:
    """Test an average agent that guesses randomly."""
    env = EmailTriageEnv(task_type=task_type, max_steps=8, dataset_mode="train")
    total_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset(seed=episode + 1000)
        episode_reward = 0.0
        done = False
        
        while not done:
            # Random action
            action_type = random.choice(["classify", "prioritize"])
            
            if action_type == "classify":
                action = Action(
                    action_type="classify",
                    label=random.choice(list(EmailCategory))
                )
            else:
                action = Action(
                    action_type="prioritize",
                    level=random.choice(list(PriorityLevel))
                )
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward.total
        
        total_rewards.append(episode_reward)
    
    return sum(total_rewards) / len(total_rewards)


def test_smart_agent(task_type: str = "easy", num_episodes: int = 30) -> float:
    """Test a smart agent that knows the ground truth (oracle)."""
    env = EmailTriageEnv(task_type=task_type, max_steps=8, dataset_mode="train")
    total_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset(seed=episode + 2000)
        episode_reward = 0.0
        done = False
        
        while not done:
            # Oracle agent knows ground truth
            current_state = env.state()
            if current_state:
                if task_type == "easy":
                    # Just classify correctly
                    action = Action(
                        action_type="classify",
                        label=current_state.ground_truth_category
                    )
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward.total
                    
                elif task_type == "medium":
                    # Just prioritize correctly
                    action = Action(
                        action_type="prioritize",
                        level=current_state.ground_truth_priority
                    )
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward.total
                    
                elif task_type == "hard":
                    # Classify then prioritize
                    action = Action(
                        action_type="classify",
                        label=current_state.ground_truth_category
                    )
                    obs, reward1, done1, info1 = env.step(action)
                    episode_reward += reward1.total
                    
                    if not done1:
                        action = Action(
                            action_type="prioritize",
                            level=current_state.ground_truth_priority
                        )
                        obs, reward2, done2, info2 = env.step(action)
                        episode_reward += reward2.total
                        done = done2
            else:
                break
        
        total_rewards.append(episode_reward)
    
    return sum(total_rewards) / len(total_rewards)


def analyze_reward_distribution(task_type: str = "easy") -> Dict[str, float]:
    """Analyze reward distribution for different agent types."""
    print(f"\n=== Analyzing Reward Balance for {task_type.upper()} Task ===")
    
    bad_reward = test_bad_agent(task_type, 30)
    avg_reward = test_average_agent(task_type, 30)
    smart_reward = test_smart_agent(task_type, 20)
    
    print(f"Bad agent (always wrong): {bad_reward:.3f}")
    print(f"Average agent (random): {avg_reward:.3f}")
    print(f"Smart agent (oracle): {smart_reward:.3f}")
    
    # Check balance criteria - updated for benchmark environment
    # Random actions should be penalized (negative), not near zero
    criteria_met = {
        "bad_negative": bad_reward < -0.5,  # Bad agents should be strongly negative
        "average_negative": avg_reward < 0,  # Random actions should be penalized
        "smart_positive": smart_reward > 0.2,
        "proper_ordering": bad_reward < avg_reward < smart_reward
    }
    
    print("\nBalance Criteria:")
    for criterion, met in criteria_met.items():
        status = "[PASS]" if met else "[FAIL]"
        print(f"  {status} {criterion}")
    
    return {
        "bad": bad_reward,
        "average": avg_reward,
        "smart": smart_reward,
        "criteria_met": criteria_met
    }


def main():
    print("=== Email Triage Environment Reward Balance Analysis ===")
    
    results = {}
    
    for task_type in ["easy", "medium", "hard"]:
        results[task_type] = analyze_reward_distribution(task_type)
    
    print("\n=== Overall Assessment ===")
    
    all_criteria_met = True
    for task_type, result in results.items():
        task_ok = all(result["criteria_met"].values())
        status = "GOOD" if task_ok else "NEEDS ADJUSTMENT"
        print(f"{task_type.upper()} task: {status}")
        
        if not task_ok:
            all_criteria_met = False
            failed = [k for k, v in result["criteria_met"].items() if not v]
            print(f"  Failed criteria: {failed}")
    
    if all_criteria_met:
        print("\n[SUCCESS] All tasks have proper reward balance!")
        print("   Bad -> negative, Average -> penalized, Smart -> positive")
    else:
        print("\n[WARNING] Reward balance needs adjustment")
        print("   Consider adjusting reward magnitudes in _evaluate_classification and _evaluate_priority")
    
    # Provide specific recommendations
    print("\n=== Recommendations ===")
    print("1. If smart agent reward is too low, increase correct action rewards")
    print("2. If average agent reward is too negative, reduce penalties for medium/hard emails")
    print("3. Ensure ordering: bad < average < smart")
    print("4. Consider task difficulty scaling (hard task should have higher smart rewards)")


if __name__ == "__main__":
    main()
