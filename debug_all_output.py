#!/usr/bin/env python3
"""Debug script to capture ALL output from inference.py and check for any problematic scores."""

import subprocess
import sys
import re
import os

def capture_all_output():
    """Run inference.py and capture all output (stdout and stderr)."""
    print("Running inference.py and capturing ALL output...")
    print("=" * 80)
    
    # Set environment variable to ensure consistent output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # Run inference.py and capture output
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        print(f"STDOUT length: {len(stdout)} characters")
        print(f"STDERR length: {len(stderr)} characters")
        
        # Combine all output
        all_output = stdout + "\n" + stderr
        
        # Write to file for inspection
        with open("debug_output.txt", "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)
        
        print("Full output written to debug_output.txt")
        
        # Check for problematic patterns in ALL output
        print("\nSearching for problematic score patterns in ALL output...")
        print("=" * 80)
        
        problematic_patterns = [
            r'score=1\.0+',
            r'score=0\.0+',
            r'score=1(?!\.\d)',  # score=1 without decimal
            r'score=0(?!\.\d)',   # score=0 without decimal
            r'score=1\.000000',
            r'score=0\.000000',
            r'Score: 1\.000',
            r'Score: 0\.000',
            r'Average Score: 1\.000',
            r'Average Score: 0\.000',
        ]
        
        lines = all_output.split('\n')
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
            
            return False
        else:
            print("OK: No problematic score patterns found in ALL output.")
            
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
                                return False
                            else:
                                print(f"  OK Line: {line.strip()} - Score {score} OK")
                        except ValueError:
                            print(f"  WARN Line: {line.strip()} - Could not parse score")
            
            return True
            
    except subprocess.TimeoutExpired:
        print("ERROR: Inference.py timed out")
        return False
    except Exception as e:
        print(f"ERROR: Error running inference.py: {e}")
        return False

if __name__ == "__main__":
    success = capture_all_output()
    if success:
        print("\n" + "=" * 80)
        print("ALL CHECKS PASSED - No problematic scores found")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("PROBLEMS FOUND - Check the output above")
        print("=" * 80)
        sys.exit(1)