import os
import re
import json
import time
import warnings
from datetime import datetime

# ============================================================================
# GLOBAL TRACKING
# ============================================================================

# Configuration: Set the number of steps per epoch
# This is used to calculate the current epoch from the step number
STEPS_PER_EPOCH = 566  # Change this value to match your training setup

# Global step counter for tracking steps when not provided in kwargs
_step_counter = 0

# ============================================================================
# ORIGINAL IMPLEMENTATIONS (COMMENTED OUT - PRESERVED FOR REFERENCE)
# ============================================================================

# def accuracy_reward(completions, assistant, **kwargs):
# ... (rest of the file)
#     """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
#     contents = [completion[0]["content"] for completion in completions]
#     solution = [a['content'] for a in assistant]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         # Try symbolic verification first
#         try:
#             answer = parse(content)
#             if float(verify(answer, parse(sol))) > 0:
#                 reward = 1.0
#         except Exception:
#             pass  # Continue to next verification method if this fails

#         # If symbolic verification failed, try string matching
#         if reward == 0.0:
#             try:
#                 # Extract answer from solution if it has think/answer tags
#                 sol_match = re.search(r"<answer>(.*?)</answer>", sol)
#                 ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

#                 # Extract answer from content if it has think/answer tags
#                 content_match = re.search(r"<answer>(.*?)</answer>", content)
#                 student_answer = content_match.group(1).strip() if content_match else content.strip()

#                 # Compare the extracted answers
#                 if student_answer == ground_truth:
#                     reward = 1.0
#             except Exception:
#                 pass  # Keep reward as 0.0 if both methods fail

#         rewards.append(reward)
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             with open(log_path, "a") as f:
#                 f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
#                 f.write(f"Content: {content}\n")
#                 f.write(f"Solution: {sol}\n")
#     return rewards


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]

# # ============================================================================
# # NEW IMPLEMENTATIONS FOR JSON/SNOMED PREDICTION TASK
# # ============================================================================

def extract_snomed_codes(text):
    """
    Extract all SNOMED CT codes from a given piece of text.
    SNOMED CT codes are typically numeric strings (6-18 digits).
    Returns a list of unique SNOMED codes found in the text.
    """
    # First, try to extract from <answer> tags if present
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    search_text = answer_match.group(1).strip() if answer_match else text
    
    # Find all numeric sequences that could be SNOMED codes
    # SNOMED CT codes are typically 6-18 digits long
    # Pattern: sequences of digits, potentially with commas/spaces around them
    snomed_pattern = r'\b\d{6,18}\b'
    matches = re.findall(snomed_pattern, search_text)
    
    # Return unique codes (as strings)
    return list(set(matches))


def snomed_sctid_reward(completions, assistant, **kwargs):
    """
    Reward function for SNOMED SCTID prediction with detailed matching logic.
    - If no SCT IDs extracted: reward = -10
    - For every wrong code: subtract a fraction of -10 (total -10 divided among wrong codes)
    - For codes with same length as ground truth: match digits position by position
      - Reward between 1-2 based on fraction of matching digits (only for best match)
      - If only length matches (no digit matches): add 1 to reward
    - If correct answer returned: add 3 to total reward
    - Only 1 code may ever add a reward (the best match or correct one)
    - Appends correct trajectories to a JSONL file on each call:
      /log/snomed_reasoning_tracking.jsonl
    """
    global _step_counter
    
    # Get step number from kwargs if available, otherwise use global counter
    current_step = kwargs.get('step', _step_counter)
    _step_counter = current_step + 1
    
    # Calculate current epoch from step number
    current_epoch = current_step // STEPS_PER_EPOCH
    
    # Extract generated strings from model outputs
    contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    # Collect all trajectories: entry_ids, content, rewards, and ground truth
    all_entry_ids = []
    all_trajectories = []
    all_rewards = []
    all_ground_truth_sct_ids = []
    all_ground_truth_labels = []
    
    for content, asst in zip(contents, assistant):
        # Parse the JSON ground truth created in the data composition script
        try:
            # The trainer passes assistant as a list of dicts with 'content' key
            gt_data = json.loads(asst['content'])
            target_sctid = str(gt_data.get('sct_id', '')).strip()
            entry_id = str(gt_data.get('id', '')).strip()
            target_label = str(gt_data.get('label', '')).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            reward = 0.0
            rewards.append(reward)
            # Still collect even if parsing fails
            all_entry_ids.append('unknown')
            all_trajectories.append(content)
            all_rewards.append(reward)
            all_ground_truth_sct_ids.append('unknown')
            all_ground_truth_labels.append('unknown')
            warnings.warn(f"[REWARD LOG] Failed to parse ground truth: {e}")
            continue
        
        # Extract SNOMED codes from the content
        extracted_codes = extract_snomed_codes(content)
        
        # If no codes extracted: reward = -10
        if len(extracted_codes) == 0:
            reward = -10.0
            rewards.append(reward)
            # Still collect even if no codes extracted
            all_entry_ids.append(entry_id)
            all_trajectories.append(content)
            all_rewards.append(reward)
            all_ground_truth_sct_ids.append(target_sctid)
            all_ground_truth_labels.append(target_label)
            continue
        
        # Initialize reward and tracking
        reward = 0.0
        correct_found = False
        
        # Check if correct answer is in the list
        if target_sctid in extracted_codes:
            # Correct answer found: add 3
            reward = 3.0
            correct_found = True
            
            # Penalize wrong codes: -1 for each wrong code
            wrong_codes = [code for code in extracted_codes if code != target_sctid]
            if wrong_codes:
                reward -= len(wrong_codes)  # Subtract 1 for each wrong code
        else:
            # No correct answer, find best matching code (only one code can contribute)
            target_length = len(target_sctid)
            same_length_codes = [code for code in extracted_codes if len(code) == target_length]
            
            best_match = None
            best_match_score = -1
            best_match_digit_matches = 0
            
            if same_length_codes:
                # Find the code with most matching digits at same positions
                for code in same_length_codes:
                    # Count matching digits at same positions
                    matching_digits = sum(1 for i in range(min(len(code), len(target_sctid))) 
                                        if code[i] == target_sctid[i])
                    if matching_digits > best_match_score:
                        best_match_score = matching_digits
                        best_match = code
                        best_match_digit_matches = matching_digits
                
                if best_match_score > 0:
                    # Calculate reward based on fraction of matching digits: between 1 and 2
                    fraction_matching = best_match_digit_matches / target_length
                    reward = 1.0 + (1.0 * fraction_matching)  # Range: 1.0 to 2.0
                else:
                    # Only length matches, no digit matches: add 1
                    reward = 1.0
            
            # Penalize all wrong codes: -1 for each wrong code
            wrong_codes = [code for code in extracted_codes if code != target_sctid]
            if wrong_codes:
                reward -= len(wrong_codes)  # Subtract 1 for each wrong code
        
        # Collect all trajectories (not just correct ones)
        all_entry_ids.append(entry_id)
        all_trajectories.append(content)
        all_rewards.append(reward)
        all_ground_truth_sct_ids.append(target_sctid)
        all_ground_truth_labels.append(target_label)
        
        rewards.append(reward)

    # Append to JSONL file for all entries
    if all_entry_ids:
        base_log_dir = "/log"
        
        # Attempt to use /log/ as requested, fallback to ./log/ if permission denied
        try:
            if not os.path.exists(base_log_dir):
                os.makedirs(base_log_dir, exist_ok=True)
        except OSError:
            base_log_dir = os.path.join(os.getcwd(), "log")
            os.makedirs(base_log_dir, exist_ok=True)
        
        log_file = os.path.join(base_log_dir, "snomed_reasoning_tracking.jsonl")
        
        try:
            # Create JSONL entry with step, epoch, entry_ids, trajectories, rewards, and ground truth
            jsonl_entry = {
                "step": current_step,
                "epoch": current_epoch,
                "entry_ids": all_entry_ids,
                "trajectories": all_trajectories,
                "rewards": all_rewards,
                "ground_truth_sct_ids": all_ground_truth_sct_ids,
                "ground_truth_labels": all_ground_truth_labels
            }
            
            # Append to JSONL file
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[REWARD LOG] Failed to append to reasoning log: {e}")

    return rewards
