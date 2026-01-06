#!/usr/bin/env python3
"""
Validate agentic-rl training data format.

Usage:
    validate_data.py <input_dir>

Checks:
- All required files exist
- Data files have correct JSON format
- Required fields are present
- Configuration is valid
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def validate_jsonl_file(file_path: Path, required_fields: List[str]) -> tuple[bool, str]:
    """Validate JSONL file format and required fields.
    
    Returns:
        (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        if not lines:
            return False, f"File is empty: {file_path}"
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON at line {i}: {e}"
            
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field '{field}' at line {i}"
                
            if "messages" in data:
                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) == 0:
                    return False, f"'messages' must be non-empty list at line {i}"
                
                if not all("role" in msg and "content" in msg for msg in messages):
                    return False, f"Each message must have 'role' and 'content' at line {i}"
        
        return True, f"Valid ({len(lines)} records)"
        
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_json_file(file_path: Path) -> tuple[bool, str]:
    """Validate JSON file format.
    
    Returns:
        (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            return False, "File is empty or contains null"
        
        return True, "Valid"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_python_file(file_path: Path, required_function: str) -> tuple[bool, str]:
    """Validate Python file has required function.
    
    Returns:
        (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if f"def {required_function}" not in content:
            return False, f"Missing required function: {required_function}"
        
        return True, f"Valid (has {required_function})"
        
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_input_dir(input_dir: Path) -> Dict[str, tuple[bool, str]]:
    """Validate all required files in input directory.
    
    Returns:
        Dictionary mapping file names to (is_valid, message) tuples
    """
    results = {}
    
    results["train.jsonl"] = validate_jsonl_file(
        input_dir / "train.jsonl",
        required_fields=["messages", "ground_truth"]
    )
    
    results["test.jsonl"] = validate_jsonl_file(
        input_dir / "test.jsonl",
        required_fields=["messages", "ground_truth"]
    )
    
    results["config.json"] = validate_json_file(input_dir / "config.json")
    
    agent_json = input_dir / "agent.json"
    agent_jsonl = input_dir / "agent.jsonl"
    
    if agent_json.exists():
        results["agent.json"] = validate_json_file(agent_json)
        try:
            with open(agent_json) as f:
                agent_data = json.load(f)
            if "model_name" not in agent_data:
                results["agent.json"] = (False, "Missing 'model_name' field")
        except:
            pass
    elif agent_jsonl.exists():
        results["agent.json"] = validate_jsonl_file(agent_jsonl, required_fields=["model_name"])
        results["agent.json"] = (results["agent.json"][0], results["agent.json"][1].replace("agent.jsonl", "agent.json/jsonl"))
    else:
        results["agent.json"] = (False, "File not found (tried agent.json and agent.jsonl)")
    
    results["judge.py"] = validate_python_file(
        input_dir / "judge.py",
        required_function="compute_reward"
    )
    
    task_goal = input_dir / "task.goal"
    if task_goal.exists():
        results["task.goal"] = (True, "Present")
    else:
        results["task.goal"] = (False, "File not found")
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: validate_data.py <input_dir>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1]).resolve()
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)
    
    print(f"Validating input directory: {input_dir}\n")
    
    results = validate_input_dir(input_dir)
    
    all_valid = True
    for filename, (is_valid, message) in results.items():
        status = "✓" if is_valid else "✗"
        print(f"{status} {filename}: {message}")
        if not is_valid:
            all_valid = False
    
    print()
    if all_valid:
        print("✓ All validations passed!")
        sys.exit(0)
    else:
        print("✗ Some validations failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
