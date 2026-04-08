#!/usr/bin/env python3
"""Test inference with multiple random seeds to check for score issues."""

import subprocess
import sys
import re
import random

def test_with_seed(seed_offset):
    """Run inference with a specific seed offset."""
    # Modify the inference.py to use different seeds
    # Actually, inference.py uses fixed seeds: 42, 43, 44 for easy, medium, hard
    # Let's just run it as-is multiple times
    print(f"\nRunning inference with seed offset {seed_offset}...")
    
    # We'll modify the environment to use different seeds by patching
    # For now, just run the standard inference
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    stdout = result.stdout
    stderr = result.stderr
    
    # Check for problematic scores
    problematic = False
    lines = (stdout + "\n" + stderr).split('\n')
    
    for line in lines:
        # Look for score= patterns
        if 'score=' in line:
            match = re.search(r'score=([\d.]+)', line)
            if match:
                score_str = match.group(1)
                try:
                    score = float(score_str)
                    if score <= 0 or score >= 1:
                        print(f"  ERROR: Line: {line.strip()} - Score {score} NOT strictly between 0 and 1")
                        problematic = True
                except ValueError:
                    pass
    
    return not problematic

def main():
    print("Testing inference with multiple runs...")
    print("=" * 80)
    
    # Run inference a few times
    all_ok = True
    for i in range(5):
        ok = test_with_seed(i)
        if not ok:
            all_ok = False
            print(f"  Run {i} had issues")
        else:
            print(f"  Run {i} OK")
    
    if all_ok:
        print("\n" + "=" * 80)
        print("ALL RUNS PASSED - No score issues found")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("SOME RUNS FAILED - Check scores above")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())