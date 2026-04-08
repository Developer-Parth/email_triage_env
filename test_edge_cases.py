#!/usr/bin/env python3
"""Test edge cases for score clamping."""

import sys
sys.path.insert(0, '.')

from email_triage_env.tasks.graders import (
    EasyTaskGrader, MediumTaskGrader, HardTaskGrader, EpisodeResult
)

def test_easy_grader():
    """Test EasyTaskGrader edge cases."""
    print("Testing EasyTaskGrader...")
    grader = EasyTaskGrader()
    
    # Test correct classification
    result = EpisodeResult(
        task_type="easy",
        total_reward=0.0,
        classification_correct=True,
        priority_correct=False,
        steps_taken=1,
        max_steps=8,
        is_classified=True,
        is_prioritized=False,
        is_replied=False,
        action_history=[]
    )
    score = grader.grade(result)
    print(f"  Correct classification score: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    assert score == 0.999, f"Expected 0.999, got {score}"
    
    # Test incorrect classification
    result.classification_correct = False
    score = grader.grade(result)
    print(f"  Incorrect classification score: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    assert score == 0.001, f"Expected 0.001, got {score}"
    
    print("  PASS")

def test_medium_grader():
    """Test MediumTaskGrader edge cases."""
    print("Testing MediumTaskGrader...")
    grader = MediumTaskGrader()
    
    # Test correct priority
    result = EpisodeResult(
        task_type="medium",
        total_reward=0.0,
        classification_correct=False,
        priority_correct=True,
        steps_taken=1,
        max_steps=8,
        is_classified=False,
        is_prioritized=True,
        is_replied=False,
        action_history=[]
    )
    score = grader.grade(result)
    print(f"  Correct priority score: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    assert score == 0.999, f"Expected 0.999, got {score}"
    
    # Test incorrect priority
    result.priority_correct = False
    score = grader.grade(result)
    print(f"  Incorrect priority score: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    assert score == 0.001, f"Expected 0.001, got {score}"
    
    print("  PASS")

def test_hard_grader():
    """Test HardTaskGrader edge cases."""
    print("Testing HardTaskGrader...")
    grader = HardTaskGrader()
    
    # Test both correct (should return 0.999 due to clamp)
    result = EpisodeResult(
        task_type="hard",
        total_reward=0.0,
        classification_correct=True,
        priority_correct=True,
        steps_taken=1,
        max_steps=8,
        is_classified=True,
        is_prioritized=True,
        is_replied=True,
        action_history=[]
    )
    score = grader.grade(result)
    print(f"  Both correct score: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    # Allow small floating point tolerance
    assert abs(score - 0.999) < 0.0001, f"Expected ~0.999, got {score}"
    
    # Test both incorrect (should return 0.001 due to clamp)
    result.classification_correct = False
    result.priority_correct = False
    score = grader.grade(result)
    print(f"  Both incorrect score: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    assert score == 0.001, f"Expected 0.001, got {score}"
    
    # Test mixed (classification correct, priority incorrect)
    result.classification_correct = True
    result.priority_correct = False
    score = grader.grade(result)
    print(f"  Classification correct, priority incorrect: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    # 0.6 * 0.999 + 0.4 * 0.001 = 0.5998
    expected = 0.6 * 0.999 + 0.4 * 0.001
    assert abs(score - expected) < 0.0001, f"Expected ~{expected}, got {score}"
    
    # Test mixed (classification incorrect, priority correct)
    result.classification_correct = False
    result.priority_correct = True
    score = grader.grade(result)
    print(f"  Classification incorrect, priority correct: {score}")
    assert 0 < score < 1, f"Score {score} not strictly between 0 and 1"
    # 0.6 * 0.001 + 0.4 * 0.999 = 0.4002
    expected = 0.6 * 0.001 + 0.4 * 0.999
    assert abs(score - expected) < 0.0001, f"Expected ~{expected}, got {score}"
    
    print("  PASS")

def test_floating_point():
    """Test floating point precision issues."""
    print("Testing floating point precision...")
    
    # Test that 0.6 * 1.0 + 0.4 * 1.0 doesn't produce exactly 1.0 due to floating point
    # Actually, 0.6 + 0.4 = 1.0 exactly in floating point
    weighted = 0.6 * 1.0 + 0.4 * 1.0
    print(f"  0.6 * 1.0 + 0.4 * 1.0 = {weighted}")
    print(f"  Is exactly 1.0? {weighted == 1.0}")
    print(f"  Is >= 1.0? {weighted >= 1.0}")
    
    # Test with different values
    weighted2 = 0.6 * 0.999 + 0.4 * 0.999
    print(f"  0.6 * 0.999 + 0.4 * 0.999 = {weighted2}")
    
    print("  PASS")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing score clamping edge cases")
    print("=" * 60)
    
    try:
        test_easy_grader()
        test_medium_grader()
        test_hard_grader()
        test_floating_point()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)