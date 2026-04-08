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
        
        # Find all [END] lines with scores
        # New format: [END] success={true|false} steps={n} rewards={r1,r2,...} score={score}
        pattern = r'\[END\] success=(true|false) steps=(\d+) rewards=([\d\.,\-]+) score=([\d\.]+)'
        matches = re.findall(pattern, output)
        
        if not matches:
            print("ERROR: No [END] lines found in output")
            return False
        
        print(f"Found {len(matches)} task results:")
        
        all_valid = True
        for success, steps, rewards, score_str in matches:
            try:
                score = float(score_str)
                if score <= 0.0 or score >= 1.0:
                    print(f"  [ERROR] success={success}: score={score} (INVALID - must be strictly between 0 and 1)")
                    all_valid = False
                elif score >= 0.999999:
                    print(f"  [WARN] success={success}: score={score} (close to 1, but OK)")
                elif score <= 0.000001:
                    print(f"  [WARN] success={success}: score={score} (close to 0, but OK)")
                else:
                    print(f"  [OK] success={success}: score={score} (valid)")
            except ValueError:
                print(f"  [ERROR] success={success}: score={score_str} (not a valid float)")
                all_valid = False
        
        # Also check for any problematic patterns
        problematic_patterns = [
            r'score=1\.0(?!\d)',      # score=1.0 not followed by digit
            r'score=1\.000000(?!\d)', # score=1.000000 not followed by digit
            r'score=0\.0(?!\d)',      # score=0.0 not followed by digit
            r'score=0\.000000(?!\d)', # score=0.000000 not followed by digit
            r'score=1[^\.\d]',        # score=1 followed by non-dot, non-digit
            r'score=0[^\.\d]',        # score=0 followed by non-dot, non-digit
        ]
        
        print("\nChecking for problematic patterns...")
        for pattern in problematic_patterns:
            if re.search(pattern, output):
                print(f"  [ERROR] Found problematic pattern: {pattern}")
                all_valid = False
        
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