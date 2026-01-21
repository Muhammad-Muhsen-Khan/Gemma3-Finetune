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

# Global dictionary to track reasoning trajectories across training steps
reasoning_dict = {"total_reasonings": 0}
# Global step counter for tracking save intervals
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
    Reward function for SNOMED SCTID prediction.
    - Extracts all SNOMED codes from the output (preferring <answer> tags if present).
    - +1.0 reward if the correct SCTID is found in the extracted codes.
    - -1.0 reward for each wrong SCTID found in the extracted codes.
    - Defaults to -1.0 reward if no <answer> tags are present.
    - Always logs raw content when the answer is correct.
    - Increments think counter only when <think> tags are found.
    - Saves reasoning dictionary every 50 steps in epoch-specific folders:
      /log/epoch_{epoch}/step_{step:05d}_snomed_reasoning_tracking.json
    """
    global reasoning_dict
    
    # Extract generated strings from model outputs
    contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    
    for content, asst in zip(contents, assistant):
        # Parse the JSON ground truth created in the data composition script
        try:
            # The trainer passes assistant as a list of dicts with 'content' key
            gt_data = json.loads(asst['content'])
            target_sctid = str(gt_data.get('sct_id', '')).strip()
            entry_id = str(gt_data.get('id', '')).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            rewards.append(0.0)
            warnings.warn(f"[REWARD LOG] Failed to parse ground truth: {e}")
            continue
        
        # Check if <answer> tags are present
        has_answer_tags = bool(re.search(r"<answer>.*?</answer>", content, re.DOTALL))
        
        # Extract SNOMED codes from the content
        extracted_codes = extract_snomed_codes(content)
        
        # Initialize reward
        reward = 0.0
        correct_found = False
        
        # If no answer tags, default to -1
        if not has_answer_tags:
            reward = -1.0
        else:
            # Check if target SCTID is in the extracted codes
            if target_sctid in extracted_codes:
                reward += 1.0
                correct_found = True
            
            # Penalize for each wrong SCTID found
            for code in extracted_codes:
                if code != target_sctid:
                    reward -= 1.0
        
        # Always log raw content when answer is correct
        if correct_found:
            if entry_id not in reasoning_dict:
                reasoning_dict[entry_id] = []
            reasoning_dict[entry_id].append(content)
            
            # Increment think counter only if <think> tags are found
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                reasoning_str = think_match.group(1).strip()
                if reasoning_str:
                    reasoning_dict["total_reasonings"] += 1
            
        rewards.append(reward)

    # 3. Persistence: Save to /log/ every 50 steps in epoch-specific folders
    global _step_counter
    
    # Get step number from kwargs if available, otherwise use global counter
    current_step = kwargs.get('step', _step_counter)
    _step_counter = current_step + 1
    
    # Calculate current epoch from step number
    current_epoch = current_step // STEPS_PER_EPOCH
    
    # Only save every 50 steps
    if current_step % 50 == 0:
        save_start = time.time()
        base_log_dir = "/log"
        
        # Attempt to use /log/ as requested, fallback to ./log/ if permission denied
        try:
            if not os.path.exists(base_log_dir):
                os.makedirs(base_log_dir, exist_ok=True)
        except OSError:
            base_log_dir = os.path.join(os.getcwd(), "log")
            os.makedirs(base_log_dir, exist_ok=True)
        
        # Create epoch-specific folder
        epoch_log_dir = os.path.join(base_log_dir, f"epoch_{current_epoch}")
        try:
            if not os.path.exists(epoch_log_dir):
                os.makedirs(epoch_log_dir, exist_ok=True)
        except OSError as e:
            print(f"[REWARD LOG] Failed to create epoch directory {epoch_log_dir}: {e}")
            return rewards
        
        # Include step number in filename: step_00050_snomed_reasoning_tracking.json
        log_file = os.path.join(epoch_log_dir, f"step_{current_step:05d}_snomed_reasoning_tracking.json")
        
        try:
            with open(log_file, "w", encoding='utf-8') as f:
                json.dump(reasoning_dict, f, ensure_ascii=False, indent=2)
            
            save_duration = time.time() - save_start
            # Print terminal warning with save duration
            print(f"[REWARD LOG] Reasoning dictionary saved to {log_file} in {save_duration:.4f} seconds. Epoch: {current_epoch}, Step: {current_step}, Total reasonings: {reasoning_dict['total_reasonings']}")
        except Exception as e:
            print(f"[REWARD LOG] Failed to save reasoning log: {e}")

    return rewards
