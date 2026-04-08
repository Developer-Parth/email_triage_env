#!/usr/bin/env python3
"""
Validate that inference.py output meets all requirements.
"""
import subprocess
import sys
import re

def validate_output():
    """Run inference.py and validate its output against all requirements."""
    print("Running inference.py and validating output...")
    
    try:
        # Run inference.py, redirect stderr to null to only capture stdout
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
        
        # Split into lines
        lines = stdout.strip().split('\n')
        print(f"\nFirst few lines of stdout:")
        for i, line in enumerate(lines[:10]):
            print(f"  {i}: {line}")
        
        # Requirement 1: Exactly 3 [START] blocks
        start_matches = re.findall(r'\[START\]', stdout)
        start_count = len(start_matches)
        print(f"\n1. Exactly 3 [START] blocks: {start_count} {'[OK]' if start_count == 3 else '[FAIL]'}")
        
        # Requirement 2: Exactly 3 [END] blocks
        end_matches = re.findall(r'\[END\]', stdout)
        end_count = len(end_matches)
        print(f"2. Exactly 3 [END] blocks: {end_count} {'[OK]' if end_count == 3 else '[FAIL]'}")
        
        # Requirement 3: Each [END] has score between 0.000001 and 0.999999
        end_pattern = r'\[END\] task=(\w+) score=([\d\.]+) steps=(\d+)'
        end_lines = re.findall(end_pattern, stdout)
        
        print(f"3. Each [END] score between 0.000001 and 0.999999:")
        all_scores_valid = True
        for task_type, score_str, steps in end_lines:
            try:
                score = float(score_str)
                if 0.000001 <= score <= 0.999999:
                    print(f"   [OK] {task_type}: score={score_str}")
                else:
                    print(f"   [FAIL] {task_type}: score={score_str} (OUT OF RANGE)")
                    all_scores_valid = False
            except ValueError:
                print(f"   [FAIL] {task_type}: score={score_str} (NOT A VALID FLOAT)")
                all_scores_valid = False
        
        # Requirement 4: No duplicate runs
        # Check that we have exactly one of each task type
        task_types = [task for task, _, _ in end_lines]
        unique_tasks = set(task_types)
        has_duplicates = len(task_types) != len(unique_tasks)
        print(f"4. No duplicate runs: {'[OK]' if not has_duplicates else '[FAIL]'}")
        if has_duplicates:
            print(f"   Found duplicate tasks: {task_types}")
        
        # Requirement 5: No extra prints before [START]
        first_line = lines[0] if lines else ""
        has_extra_before_start = not first_line.startswith('[START]')
        print(f"5. No extra prints before [START]: {'[OK]' if not has_extra_before_start else '[FAIL]'}")
        if has_extra_before_start:
            print(f"   First line is: '{first_line}'")
        
        # Requirement 6: No scientific notation
        scientific_pattern = r'[0-9]+\.[0-9]+e[+-][0-9]+'
        has_scientific = bool(re.search(scientific_pattern, stdout))
        print(f"6. No scientific notation: {'[OK]' if not has_scientific else '[FAIL]'}")
        if has_scientific:
            matches = re.findall(scientific_pattern, stdout)
            print(f"   Found scientific notation: {matches}")
        
        # Requirement 7: No hidden logs leaking into stdout
        # Check for any lines that don't match expected patterns
        expected_patterns = [
            r'^\[START\]',
            r'^\[STEP\]',
            r'^\[END\]'
        ]
        
        unexpected_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not any(re.match(pattern, line) for pattern in expected_patterns):
                unexpected_lines.append(line)
        
        has_hidden_logs = len(unexpected_lines) > 0
        print(f"7. No hidden logs leaking into stdout: {'[OK]' if not has_hidden_logs else '[FAIL]'}")
        if has_hidden_logs:
            print(f"   Unexpected lines: {unexpected_lines}")
        
        # Summary
        all_passed = (
            start_count == 3 and
            end_count == 3 and
            all_scores_valid and
            not has_duplicates and
            not has_extra_before_start and
            not has_scientific and
            not has_hidden_logs
        )
        
        print(f"\n{'='*60}")
        if all_passed:
            print("[OK] ALL REQUIREMENTS SATISFIED! Ready for submission.")
            return True
        else:
            print("[FAIL] SOME REQUIREMENTS FAILED. Do not submit yet.")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: inference.py timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = validate_output()
    sys.exit(0 if success else 1)