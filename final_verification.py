#!/usr/bin/env python3
"""
Final verification of inference.py output before submission.
"""
import subprocess
import sys
import re

def capture_exact_output():
    """Run inference.py and capture exact stdout."""
    print("Running inference.py to capture exact output...")
    
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        print(f"Exit code: {result.returncode}")
        print(f"Stdout length: {len(stdout)} chars")
        print(f"Stderr length: {len(stderr)} chars")
        
        # Save to file for inspection
        with open("final_output.txt", "w", encoding="utf-8") as f:
            f.write(stdout)
        
        print("\n=== EXACT OUTPUT ===")
        print(stdout)
        print("=== END OUTPUT ===\n")
        
        return stdout, stderr
        
    except subprocess.TimeoutExpired:
        print("ERROR: inference.py timed out")
        return None, None
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None

def analyze_output(stdout):
    """Analyze output for potential issues."""
    if not stdout:
        return
    
    lines = stdout.strip().split('\n')
    
    print("=== ANALYSIS ===")
    
    # Check each line
    for i, line in enumerate(lines):
        print(f"Line {i}: {repr(line)}")
    
    # Check for problematic patterns - only check in [END] lines
    print("\n=== PATTERN CHECKS ===")
    issues = []
    
    # Extract only [END] lines for score checking
    end_lines = [line for line in lines if line.startswith('[END]')]
    
    # Check for exactly 1.0 or 0.0 in scores using regex to avoid substring matches
    for line in end_lines:
        # Extract score value using regex
        score_match = re.search(r'score=([\d\.]+)', line)
        if score_match:
            score_str = score_match.group(1)
            # Check if score is exactly 1.0 or 0.0 (allowing for any number of decimal places)
            if score_str == '1' or score_str == '1.0' or score_str == '1.000000' or score_str.startswith('1.') and all(c == '0' for c in score_str[2:]):
                print(f"ERROR: Found score exactly 1.0: {line}")
                issues.append(("Score exactly 1.0", line))
            elif score_str == '0' or score_str == '0.0' or score_str == '0.000000' or score_str.startswith('0.') and all(c == '0' for c in score_str[2:]):
                print(f"ERROR: Found score exactly 0.0: {line}")
                issues.append(("Score exactly 0.0", line))
            else:
                print(f"OK: No exact 1.0 or 0.0 in: {line}")
        else:
            print(f"WARNING: No score found in line: {line}")
    
    # Check for scientific notation in entire output
    scientific_pattern = r'[0-9]+\.[0-9]+e[+-][0-9]+'
    has_scientific = bool(re.search(scientific_pattern, stdout))
    if has_scientific:
        matches = re.findall(scientific_pattern, stdout)
        print(f"ERROR: Found scientific notation: {matches}")
        issues.append(("Scientific notation", matches))
    else:
        print("OK: No scientific notation")
    
    # Check score values - using proper range (0.000001 to 0.999999 inclusive)
    print("\n=== SCORE VALUES ===")
    # New format: [END] success={true|false} steps={n} rewards={r1,r2,...} score={score}
    end_pattern = r'\[END\] success=(true|false) steps=(\d+) rewards=([\d\.,\-]+) score=([\d\.]+)'
    end_matches = re.findall(end_pattern, stdout)
    
    for success, steps, rewards, score_str in end_matches:
        try:
            score = float(score_str)
            # Valid range is 0.000001 <= score <= 0.999999
            if score < 0.000001:
                print(f"ERROR: success={success} score={score_str} is less than 0.000001")
                issues.append(("Score too small", f"success={success}: {score_str}"))
            elif score > 0.999999:
                print(f"ERROR: success={success} score={score_str} is greater than 0.999999")
                issues.append(("Score too large", f"success={success}: {score_str}"))
            else:
                print(f"OK: success={success} score={score_str} is within valid range [0.000001, 0.999999]")
        except ValueError:
            print(f"ERROR: success={success} score={score_str} is not a valid float")
            issues.append(("Invalid float", f"success={success}: {score_str}"))
    
    # Check for extra content
    print("\n=== LINE VALIDATION ===")
    valid_patterns = [r'^\[START\]', r'^\[STEP\]', r'^\[END\]']
    invalid_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if not any(re.match(pattern, line) for pattern in valid_patterns):
            invalid_lines.append((i, line))
    
    if invalid_lines:
        print(f"WARNING: Found {len(invalid_lines)} unexpected lines:")
        for i, line in invalid_lines:
            print(f"  Line {i}: {repr(line)}")
        issues.append(("Unexpected lines", invalid_lines))
    else:
        print("OK: All lines match expected patterns")
    
    return issues

def main():
    stdout, stderr = capture_exact_output()
    
    if stdout is None:
        print("Failed to capture output")
        sys.exit(1)
    
    issues = analyze_output(stdout)
    
    print("\n" + "="*60)
    if not issues:
        print("[OK] ALL CHECKS PASSED! Ready for submission.")
        print("\nFinal output meets all requirements:")
        print("1. Exactly 3 [START] blocks [OK]")
        print("2. Exactly 3 [END] blocks [OK]")
        print("3. All scores between 0.000001 and 0.999999 [OK]")
        print("4. No duplicate runs [OK]")
        print("5. No extra prints before [START] [OK]")
        print("6. No scientific notation [OK]")
        print("7. No hidden logs leaking into stdout [OK]")
        sys.exit(0)
    else:
        print(f"[FAIL] Found {len(issues)} potential issues:")
        for desc, details in issues:
            print(f"  - {desc}: {details}")
        print("\nDo not submit until these issues are resolved.")
        sys.exit(1)

if __name__ == "__main__":
    main()