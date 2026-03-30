#!/usr/bin/env python3
"""
Final demonstration of the Email Triage Benchmark environment.
Shows the complete capabilities including:
1. Multi-step decision making
2. Ambiguity handling
3. Efficiency optimization
4. State-based traps
5. Generalization testing
"""

import sys
import random
from typing import Dict, Any

sys.path.insert(0, '.')

from email_triage_env.models import Action, EmailCategory, PriorityLevel
from email_triage_env.environment import EmailTriageEnv


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_multi_step_reasoning():
    """Demonstrate multi-step reasoning in hard task."""
    print_header("1. Multi-Step Reasoning Demonstration")
    
    env = EmailTriageEnv(task_type="hard", max_steps=8, dataset_mode="train")
    obs = env.reset(seed=42)
    
    print(f"Email: {obs.subject}")
    print(f"Body preview: {obs.body[:100]}...")
    print(f"Step count: {obs.step_count}")
    
    # Step 1: Classify correctly
    print("\n--- Step 1: Classification ---")
    action1 = Action(action_type="classify", label=EmailCategory.WORK)
    obs, reward1, done1, info1 = env.step(action1)
    print(f"Action: classify as WORK")
    classification_val = reward1.classification if reward1.classification is not None else 0.0
    step_cost_val = reward1.step_cost if reward1.step_cost is not None else 0.0
    print(f"Reward: {reward1.total:.2f} (classification: {classification_val:.2f}, step_cost: {step_cost_val:.2f})")
    print(f"Info: {info1.get('sequence_quality', 'N/A')}")
    
    # Step 2: Prioritize correctly
    print("\n--- Step 2: Prioritization ---")
    action2 = Action(action_type="prioritize", level=PriorityLevel.MEDIUM)
    obs, reward2, done2, info2 = env.step(action2)
    print(f"Action: prioritize as MEDIUM")
    priority_val = reward2.priority if reward2.priority is not None else 0.0
    step_cost_val2 = reward2.step_cost if reward2.step_cost is not None else 0.0
    print(f"Reward: {reward2.total:.2f} (priority: {priority_val:.2f}, step_cost: {step_cost_val2:.2f})")
    
    # Step 3: Generate reply
    print("\n--- Step 3: Reply Generation ---")
    action3 = Action(action_type="reply", text="I'll review this and get back to you by EOD.")
    obs, reward3, done3, info3 = env.step(action3)
    print(f"Action: reply with text")
    sequence_val = reward3.sequence if reward3.sequence is not None else 0.0
    completion_val = reward3.completion_bonus if reward3.completion_bonus is not None else 0.0
    print(f"Reward: {reward3.total:.2f} (sequence: {sequence_val:.2f}, completion_bonus: {completion_val:.2f})")
    
    print(f"\nTotal reward: {reward1.total + reward2.total + reward3.total:.2f}")
    print(f"Task completed: {done3}")
    state_obj = env.state()
    if state_obj:
        print(f"State: Classified: {state_obj.is_classified}, Prioritized: {state_obj.is_prioritized}, Replied: {state_obj.is_replied}")
    else:
        print(f"State: Not available")


def demo_ambiguity_handling():
    """Demonstrate handling of ambiguous edge cases."""
    print_header("2. Ambiguity Handling Demonstration")
    
    # Test with the "insane edge cases"
    edge_cases = [
        "SECURITY ALERT: Unusual login detected from New York",
        "Hey, got a minute?",
        "TIME-SENSITIVE: Quarterly planning document",
        "Your health insurance renewal"
    ]
    
    for i, subject in enumerate(edge_cases[:2]):  # Show first 2 for brevity
        print(f"\n--- Edge Case {i+1}: '{subject}' ---")
        
        # Create environment and find the email
        env = EmailTriageEnv(task_type="easy", max_steps=8, dataset_mode="test")
        
        # We need to find this email in the dataset
        # For demo purposes, we'll just show the challenge
        print("Challenge: This email contains mixed signals:")
        if "SECURITY" in subject:
            print("  - Looks like spam/phishing (urgent tone, security claim)")
            print("  - But could be legitimate security alert from IT")
            print("  - Requires careful contextual analysis")
        elif "Hey" in subject:
            print("  - Casual tone suggests personal email")
            print("  - But could be from colleague about work matter")
            print("  - Boundary between work/personal is blurred")
        
        # Show what a smart agent might do
        print("\nSmart agent approach:")
        print("  1. Analyze sender, content, and context")
        print("  2. Consider organizational policies")
        print("  3. Make conservative classification")
        print("  4. Assign appropriate priority based on content")


def demo_efficiency_optimization():
    """Demonstrate efficiency optimization with step costs."""
    print_header("3. Efficiency Optimization Demonstration")
    
    env = EmailTriageEnv(task_type="hard", max_steps=8, dataset_mode="train")
    obs = env.reset(seed=123)
    
    print("Email: Team meeting rescheduled")
    print("Goal: Complete triage with minimal steps")
    
    # Inefficient approach (extra steps)
    print("\n--- Inefficient Approach (4 steps) ---")
    env_ineff = EmailTriageEnv(task_type="hard", max_steps=8, dataset_mode="train")
    obs = env_ineff.reset(seed=123)
    
    steps = [
        ("classify", EmailCategory.WORK),
        ("prioritize", PriorityLevel.LOW),  # Wrong priority
        ("prioritize", PriorityLevel.MEDIUM),  # Correct it
        ("reply", "Noted, will attend.")
    ]
    
    total_reward = 0
    for i, (action_type, value) in enumerate(steps):
        if action_type == "classify":
            action = Action(action_type="classify", label=value)
        elif action_type == "prioritize":
            action = Action(action_type="prioritize", level=value)
        else:
            action = Action(action_type="reply", text=value)
        
        obs, reward, done, info = env_ineff.step(action)
        total_reward += reward.total
        print(f"  Step {i+1}: {action_type} -> reward: {reward.total:.2f} (step_cost: {reward.step_cost:.2f})")
    
    print(f"  Total reward: {total_reward:.2f}")
    
    # Efficient approach (3 steps)
    print("\n--- Efficient Approach (3 steps) ---")
    env_eff = EmailTriageEnv(task_type="hard", max_steps=8, dataset_mode="train")
    obs = env_eff.reset(seed=123)
    
    steps_eff = [
        ("classify", EmailCategory.WORK),
        ("prioritize", PriorityLevel.MEDIUM),  # Correct first time
        ("reply", "Noted, will attend.")
    ]
    
    total_reward_eff = 0
    for i, (action_type, value) in enumerate(steps_eff):
        if action_type == "classify":
            action = Action(action_type="classify", label=value)
        elif action_type == "prioritize":
            action = Action(action_type="prioritize", level=value)
        else:
            action = Action(action_type="reply", text=value)
        
        obs, reward, done, info = env_eff.step(action)
        total_reward_eff += reward.total
        print(f"  Step {i+1}: {action_type} -> reward: {reward.total:.2f} (step_cost: {reward.step_cost:.2f})")
    
    print(f"  Total reward: {total_reward_eff:.2f}")
    print(f"\nEfficiency gain: {total_reward_eff - total_reward:.2f} (saved 1 step)")


def demo_state_based_traps():
    """Demonstrate state-based traps for consistency."""
    print_header("4. State-Based Traps Demonstration")
    
    env = EmailTriageEnv(task_type="easy", max_steps=8, dataset_mode="train")
    obs = env.reset(seed=99)
    
    print("Testing consistency penalties for erratic behavior")
    print(f"Initial email: {obs.subject}")
    
    # Make correct classification
    print("\n--- Step 1: Correct classification ---")
    action1 = Action(action_type="classify", label=EmailCategory.PERSONAL)
    obs, reward1, done1, info1 = env.step(action1)
    print(f"Action: classify as PERSONAL")
    print(f"Reward: {reward1.total:.2f}")
    
    # Change to wrong classification (inconsistent)
    print("\n--- Step 2: Change to wrong classification (inconsistent) ---")
    action2 = Action(action_type="classify", label=EmailCategory.SPAM)
    obs, reward2, done2, info2 = env.step(action2)
    print(f"Action: change to SPAM (incorrect)")
    print(f"Reward: {reward2.total:.2f}")
    state_traps_val2 = reward2.state_traps if reward2.state_traps is not None else 0.0
    print(f"State trap penalty: {state_traps_val2:.2f} (for changing correct decision)")
    
    # Repeat same action type
    print("\n--- Step 3: Repeat classification again ---")
    action3 = Action(action_type="classify", label=EmailCategory.WORK)
    obs, reward3, done3, info3 = env.step(action3)
    print(f"Action: classify again as WORK")
    print(f"Reward: {reward3.total:.2f}")
    state_traps_val3 = reward3.state_traps if reward3.state_traps is not None else 0.0
    print(f"State trap penalty: {state_traps_val3:.2f} (for repeating action type)")
    
    print("\nSummary: State traps penalize inconsistent behavior")
    print("  - Changing correct decisions: -0.1 penalty")
    print("  - Repeating actions excessively: -0.03 per repeat after 2")
    print("  - Encourages stable, thoughtful decision-making")


def demo_generalization_testing():
    """Demonstrate generalization testing with train/test splits."""
    print_header("5. Generalization Testing Demonstration")
    
    print("Dataset configuration:")
    print("  - Train split: 10 emails (for training/development)")
    print("  - Test split: 10 emails (for evaluation, includes edge cases)")
    print("  - Total: 20 unique emails")
    
    # Show stats for both splits
    env_train = EmailTriageEnv(task_type="easy", max_steps=8, dataset_mode="train")
    env_test = EmailTriageEnv(task_type="easy", max_steps=8, dataset_mode="test")
    
    # Get dataset stats (we need to access the dataset)
    print("\nDataset statistics:")
    
    # Create environments to get email samples
    print("\nSample from train split:")
    env_train.reset(seed=1)
    train_state = env_train.state()
    if train_state:
        print(f"  Email: {train_state.subject}")
        print(f"  Category: {train_state.ground_truth_category}")
        print(f"  Priority: {train_state.ground_truth_priority}")
    
    print("\nSample from test split (edge case):")
    env_test.reset(seed=50)  # Higher seed to get different email
    test_state = env_test.state()
    if test_state:
        print(f"  Email: {test_state.subject}")
        print(f"  Category: {test_state.ground_truth_category}")
        print(f"  Priority: {test_state.ground_truth_priority}")
        if "SECURITY" in test_state.subject or "Hey" in test_state.subject:
            print("  Note: This is an 'insane edge case' for generalization testing")
    
    print("\nGeneralization testing ensures agents:")
    print("  1. Don't memorize specific emails")
    print("  2. Learn general patterns and heuristics")
    print("  3. Handle unseen, realistic edge cases")


def demo_benchmark_summary():
    """Show benchmark summary and key metrics."""
    print_header("6. Benchmark Summary & Key Metrics")
    
    print("The Email Triage Benchmark evaluates agents on 5 key dimensions:")
    
    metrics = [
        ("Decision Quality", "Accuracy in classification and prioritization", "0.0-1.0 score"),
        ("Ambiguity Handling", "Performance on edge cases with mixed signals", "Success rate on test split"),
        ("Efficiency", "Steps taken to complete tasks", "Step count (lower is better)"),
        ("Consistency", "Stability of decisions across steps", "State trap penalties (lower is better)"),
        ("Generalization", "Performance on unseen emails", "Train vs test performance gap")
    ]
    
    for name, description, metric in metrics:
        print(f"\n{name}:")
        print(f"  {description}")
        print(f"  Metric: {metric}")
    
    print("\n" + "-" * 50)
    print("Expected Performance Ranges:")
    print("-" * 50)
    print("Smart agents (oracle):    0.4 - 1.2 total reward")
    print("Average agents (random):  -1.0 - -3.0 total reward")
    print("Bad agents (always wrong): -4.0 - -6.0 total reward")
    print("Exploit resistance:       Negative rewards for simple strategies")
    
    print("\n" + "-" * 50)
    print("Dataset Characteristics:")
    print("-" * 50)
    print("Total emails: 20 (10 train, 10 test)")
    print("Categories: WORK, PERSONAL, SPAM")
    print("Priorities: LOW, MEDIUM, HIGH")
    print("Edge cases: 4 'insane' realistic tricky emails")
    print("Difficulty: Mix of easy, medium, hard emails")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("EMAIL TRIAGE BENCHMARK - FINAL DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstration showcases the complete capabilities of the")
    print("Email Triage Benchmark environment for evaluating LLM agents.")
    
    demo_multi_step_reasoning()
    demo_ambiguity_handling()
    demo_efficiency_optimization()
    demo_state_based_traps()
    demo_generalization_testing()
    demo_benchmark_summary()
    
    print_header("DEMONSTRATION COMPLETE")
    print("\nThe Email Triage Benchmark is ready for:")
    print("1. Research on multi-step decision-making")
    print("2. Evaluation of LLM agent capabilities")
    print("3. Benchmarking different AI architectures")
    print("4. Studying human-like reasoning under ambiguity")
    
    print("\nTo use the benchmark:")
    print("  python inference.py                    # Run baseline evaluation")
    print("  python -m email_triage_env.server     # Start OpenEnv server")
    print("  docker build -t email-triage .        # Build Docker image")
    print("  docker run -p 8000:8000 email-triage  # Run container")
    
    print("\nSee README.md for complete documentation and API reference.")


if __name__ == "__main__":
    main()
