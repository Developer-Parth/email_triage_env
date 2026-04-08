#!/usr/bin/env python3
"""Debug script to run inference and check all score outputs."""

import subprocess
import sys
import re

def run_inference_and_check():
    """Run inference.py and check all score outputs."""
    print("Running inference.py to capture all score outputs...")
    print("=" * 80)
    
    # Run inference.py and capture output
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        print("STDOUT (what validator sees):")
        print("-" * 80)
        print(stdout[:2000])  # First 2000 chars
        print("-" * 80)
        
        print("\nSTDERR (diagnostic output):")
        print("-" * 80)
        print(stderr[:1000])  # First 1000 chars
        print("-" * 80)
        
        # Check for problematic score patterns
        print("\nSearching for problematic score patterns...")
        print("=" * 80)
        
        # Patterns that would fail validation
        problematic_patterns = [
            r'score=1\.0+',
            r'score=0\.0+',
            r'score=1(?!\.\d)',  # score=1 without decimal
            r'score=0(?!\.\d)',   # score=0 without decimal
            r'score=1\.000000',
            r'score=0\.000000',
        ]
        
        all_lines = stdout + "\n" + stderr
        lines = all_lines.split('\n')
        
        issues = []
        for i, line in enumerate(lines, 1):
            for pattern in problematic_patterns:
                if re.search(pattern, line):
                    issues.append(f"Line {i}: {line.strip()}")
                    break
        
        if issues:
            print("ERROR: PROBLEMATIC SCORES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
            
            # Also check for [END] lines specifically
            print("\n[END] lines (what validator parses):")
            for line in lines:
                if '[END]' in line:
                    print(f"  {line.strip()}")
        else:
            print("OK: No problematic score patterns found in output.")
            
            # Still show [END] lines for verification
            print("\n[END] lines found:")
            for line in lines:
                if '[END]' in line:
                    print(f"  {line.strip()}")
                    
            # Check score ranges in [END] lines
            print("\nChecking score ranges in [END] lines:")
            for line in lines:
                if '[END]' in line:
                    match = re.search(r'score=([\d.]+)', line)
                    if match:
                        score_str = match.group(1)
                        try:
                            score = float(score_str)
                            if score <= 0 or score >= 1:
                                print(f"  ERROR Line: {line.strip()} - Score {score} NOT strictly between 0 and 1")
                            else:
                                print(f"  OK Line: {line.strip()} - Score {score} OK")
                        except ValueError:
                            print(f"  WARN Line: {line.strip()} - Could not parse score")
        
        return len(issues) == 0
        
    except subprocess.TimeoutExpired:
        print("ERROR: Inference.py timed out")
        return False
    except Exception as e:
        print(f"ERROR: Error running inference.py: {e}")
        return False

if __name__ == "__main__":
    success = run_inference_and_check()
    sys.exit(0 if success else 1)