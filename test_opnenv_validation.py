#!/usr/bin/env python3
"""
Quick validation of OpenEnv configuration and server.
"""

import sys
import yaml
import json
from pathlib import Path

sys.path.insert(0, '.')

def validate_opnenv_config():
    """Validate the openenv.yaml configuration file."""
    print("=== Validating OpenEnv Configuration ===")
    
    config_path = Path("openenv.yaml")
    if not config_path.exists():
        print("[ERROR] openenv.yaml not found")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_fields = ['name', 'version', 'description', 'spec']
        for field in required_fields:
            if field not in config:
                print(f"[ERROR] Missing required field: {field}")
                return False
        
        print(f"  Name: {config['name']}")
        print(f"  Version: {config['version']}")
        print(f"  Description: {config['description'][:80]}...")
        
        # Check spec structure
        spec = config['spec']
        if 'observation_space' not in spec:
            print("[ERROR] Missing observation_space in spec")
            return False
        
        if 'action_space' not in spec:
            print("[ERROR] Missing action_space in spec")
            return False
        
        print("  [PASS] Configuration structure is valid")
        return True
        
    except yaml.YAMLError as e:
        print(f"[ERROR] YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def validate_server_import():
    """Validate that the server can be imported and has required endpoints."""
    print("\n=== Validating Server Import ===")
    
    try:
        from email_triage_env.server import app
        
        # Check required routes
        routes = [route.path for route in app.routes]
        required_routes = ['/step', '/reset', '/state']
        
        print(f"  Found {len(routes)} routes")
        
        for route in required_routes:
            if route not in routes:
                # Check for route with methods
                route_found = False
                for r in app.routes:
                    if r.path == route:
                        route_found = True
                        break
                
                if not route_found:
                    print(f"  [WARNING] Route {route} not found in server routes")
                else:
                    print(f"  [PASS] Route {route} found")
            else:
                print(f"  [PASS] Route {route} found")
        
        print("  [PASS] Server import successful")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Failed to import server: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error during server import: {e}")
        return False


def validate_models():
    """Validate that Pydantic models can be instantiated."""
    print("\n=== Validating Pydantic Models ===")
    
    try:
        from email_triage_env.models import (
            Observation, Action, Reward, State,
            EmailCategory, PriorityLevel
        )
        
        # Test enum values
        print(f"  Email categories: {[e.value for e in EmailCategory]}")
        print(f"  Priority levels: {[p.value for p in PriorityLevel]}")
        
        # Test model creation
        obs = Observation(
            email_id="test_001",
            subject="Test Email",
            body="This is a test email body.",
            step_count=0
        )
        print(f"  Observation created: {obs.email_id}")
        
        action = Action(
            action_type="classify",
            label=EmailCategory.WORK
        )
        print(f"  Action created: {action.action_type}")
        
        reward = Reward(
            total=0.5,
            classification=0.4,
            step_cost=-0.05
        )
        print(f"  Reward created: {reward.total}")
        
        state = State(
            email_id="test_001",
            subject="Test Email",
            body="Test body",
            ground_truth_category=EmailCategory.WORK,
            ground_truth_priority=PriorityLevel.MEDIUM
        )
        print(f"  State created: {state.email_id}")
        
        print("  [PASS] All models can be instantiated")
        return True
        
    except Exception as e:
        print(f"[ERROR] Model validation failed: {e}")
        return False


def validate_dockerfile():
    """Validate Dockerfile structure."""
    print("\n=== Validating Dockerfile ===")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("[ERROR] Dockerfile not found")
        return False
    
    try:
        with open(dockerfile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required instructions
        required_keywords = ['FROM', 'COPY', 'RUN', 'CMD', 'EXPOSE']
        found_keywords = []
        
        for keyword in required_keywords:
            if keyword in content:
                found_keywords.append(keyword)
        
        print(f"  Found Docker instructions: {', '.join(found_keywords)}")
        
        if len(found_keywords) < 4:
            print(f"  [WARNING] Some required Docker instructions missing")
        else:
            print("  [PASS] Dockerfile has basic structure")
        
        # Check port exposure
        if 'EXPOSE 8000' in content:
            print("  [PASS] Port 8000 exposed correctly")
        else:
            print("  [WARNING] Port 8000 not exposed or different port used")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Dockerfile validation failed: {e}")
        return False


def main():
    """Run all validations."""
    print("OpenEnv Environment Validation")
    print("=" * 50)
    
    results = []
    
    results.append(("Configuration", validate_opnenv_config()))
    results.append(("Server Import", validate_server_import()))
    results.append(("Pydantic Models", validate_models()))
    results.append(("Dockerfile", validate_dockerfile()))
    
    print("\n" + "=" * 50)
    print("Validation Summary:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20} [{status}]")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("[SUCCESS] All validations passed!")
        print("The Email Triage Benchmark is ready for deployment.")
    else:
        print("[WARNING] Some validations failed.")
        print("Check the errors above before deployment.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)