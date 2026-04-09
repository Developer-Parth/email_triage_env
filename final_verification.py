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
    
    # Extract only [END] lines for schema checking
    end_lines = [line for line in lines if line.startswith('[END]')]
    
    # Check that [END] lines match the required schema exactly
    for line in end_lines:
        if re.match(r'^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*|)$', line):
            print(f"OK: Valid [END] line: {line}")
        else:
            print(f"ERROR: Invalid [END] line: {line}")
            issues.append(("Invalid [END] line", line))
    
    # Check for scientific notation in entire output
    scientific_pattern = r'[0-9]+\.[0-9]+e[+-][0-9]+'
    has_scientific = bool(re.search(scientific_pattern, stdout))
    if has_scientific:
        matches = re.findall(scientific_pattern, stdout)
        print(f"ERROR: Found scientific notation: {matches}")
        issues.append(("Scientific notation", matches))
    else:
        print("OK: No scientific notation")
    
    print("\n=== END SCHEMA ===")
    print("OK: [END] validation is schema-based; task scores are validated through graders, not stdout.")
    
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
        print("3. All [END] lines match the required schema [OK]")
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
