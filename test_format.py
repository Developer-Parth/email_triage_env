#!/usr/bin/env python3
import subprocess
import sys
import re
import os

def test_format():
    # Run inference.py
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=30
    )
    
    stdout = result.stdout
    lines = stdout.strip().split('\n')
    
    print("Testing output format...")
    
    # Check [START] format
    start_lines = [line for line in lines if line.startswith('[START]')]
    if len(start_lines) != 3:
        print(f"FAIL: Expected 3 [START] lines, got {len(start_lines)}")
        return False
    
    for line in start_lines:
        if not re.match(r'^\[START\] task=(easy|medium|hard) env=email-triage model=.+$', line):
            print(f"FAIL: Invalid [START] line: {line}")
            return False
    
    # Check [STEP] format
    step_lines = [line for line in lines if line.startswith('[STEP]')]
    for line in step_lines:
        if not re.match(r'^\[STEP\] step=\d+ action=\w+:\w+ reward=-?\d+\.\d{2} done=(true|false) error=null$', line):
            print(f"FAIL: Invalid [STEP] line: {line}")
            return False
    
    # Check [END] format
    end_lines = [line for line in lines if line.startswith('[END]')]
    if len(end_lines) != 3:
        print(f"FAIL: Expected 3 [END] lines, got {len(end_lines)}")
        return False
    
    for line in end_lines:
        if not re.match(r'^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*|)$', line):
            print(f"FAIL: Invalid [END] line: {line}")
            return False
    
    print("PASS: All format checks passed")
    return True

if __name__ == "__main__":
    test_format()
