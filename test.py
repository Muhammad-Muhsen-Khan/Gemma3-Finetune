#!/usr/bin/env python3
"""
Standalone test script for the SNOMED SCTID reward function.
Usage:
    python test.py "completion text" "target_sctid" "entry_id"
    python test.py "<answer>SCTID: 12345678</answer>" "12345678" "000000000001"
"""

import sys
import json
from src.train.reward_funcs import snomed_sctid_reward, extract_snomed_codes

def test_reward(completion_text, target_sctid, entry_id="000000000001"):
    """
    Test the reward function with a given completion text and target SCTID.
    
    Args:
        completion_text: The model's completion text (may include <answer> tags)
        target_sctid: The target/correct SCTID
        entry_id: Optional entry ID (default: "000000000001")
    """
    # Create the assistant data structure (as the trainer would)
    assistant_data = {
        "sct_id": target_sctid,
        "id": entry_id
    }
    assistant = [{"content": json.dumps(assistant_data, ensure_ascii=False)}]
    
    # Create completions structure (as the trainer would)
    completions = [[{"content": completion_text}]]
    
    # Call the reward function
    rewards = snomed_sctid_reward(completions, assistant, step=0)
    
    # Extract codes for display
    extracted_codes = extract_snomed_codes(completion_text)
    
    # Print results
    print("=" * 80)
    print("REWARD FUNCTION TEST")
    print("=" * 80)
    print(f"Completion text: {completion_text}")
    print(f"Target SCTID: {target_sctid}")
    print(f"Entry ID: {entry_id}")
    print(f"Extracted codes: {extracted_codes}")
    print(f"Reward: {rewards[0]}")
    print("=" * 80)
    
    return rewards[0]

if __name__ == "__main__":
    completion_text = "<answer>SCTID: 386388007 | SNOMED Description: Feeding difficulties, newborn</answer>"
    target_sctid = "311311111"
    entry_id = "000000000001"
    
    test_reward(completion_text, target_sctid, entry_id)
