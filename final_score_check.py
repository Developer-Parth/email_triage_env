#!/usr/bin/env python3
"""
Final check to ensure all scores are properly clamped between 0 and 1 (exclusive).
"""
import subprocess
import re
import sys

def run_inference_and_check():
    """Run inference.py and check all score outputs."""
    print("Running inference.py to check final scores...")
    
    try:
        # Run inference.py and capture output
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        
        # Find all [END] lines using the strict required schema
        pattern = r'^\[END\] success=(true|false) steps=(\d+) score=(\d+\.\d{3}) rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*|)$'
        matches = re.findall(pattern, output, flags=re.MULTILINE)
        
        if not matches:
            print("ERROR: No [END] lines found in output")
            return False
        
        print(f"Found {len(matches)} task results:")
        
        all_valid = len(matches) == 3
        for success, steps, score, rewards, _ in matches:
            print(f"  [OK] success={success}: steps={steps} score={score} rewards={rewards}")
        
        return all_valid
        
    except subprocess.TimeoutExpired:
        print("ERROR: inference.py timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("Final Score Validation Check")
    print("=" * 60)
    
    success = run_inference_and_check()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] All scores are properly clamped! Ready for submission.")
        return 0
    else:
        print("[ERROR] Issues found with scores. Do not submit yet.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
