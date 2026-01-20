import os
import re
import json
import time
import warnings
from datetime import datetime

# ============================================================================
# GLOBAL TRACKING
# ============================================================================

# Global dictionary to track reasoning trajectories across training steps
reasoning_dict = {"total_reasonings": 0}

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

def snomed_sctid_reward(completions, assistant, **kwargs):
    """
    Reward function for SNOMED SCTID prediction.
    - +1.0 reward if the ground truth SCTID is found anywhere in the output.
    - -1.0 reward if the ground truth SCTID is missing.
    - Logs reasoning processes (<think> tags) only if the SCTID prediction is correct.
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
        except (json.JSONDecodeError, KeyError, TypeError):
            rewards.append(0.0)
            warnings.warn(f"[REWARD LOG] Failed to parse ground truth: {e}")
            continue
            
        # 1. SCTID Reward Logic: Check if target SCTID is anywhere in the output
        if target_sctid and target_sctid in content:
            reward = 1.0
            
            # 2. Reasoning Tracking Logic (Only if the prediction is correct)
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                reasoning_str = think_match.group(1).strip()
                if reasoning_str:
                    # Increment global counter
                    reasoning_dict["total_reasonings"] += 1
                    
                    # Store reasoning per unique entry ID
                    if entry_id not in reasoning_dict:
                        reasoning_dict[entry_id] = []
                    reasoning_dict[entry_id].append(content)
        else:
            reward = -1.0
            
        rewards.append(reward)

    # 3. Persistence: Save to /log/ and time the operation
    save_start = time.time()
    log_dir = "/log"
    
    # Attempt to use /log/ as requested, fallback to ./log/ if permission denied
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    except OSError:
        log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(log_dir, exist_ok=True)
            
    log_file = os.path.join(log_dir, "snomed_reasoning_tracking.json")
    
    try:
        with open(log_file, "w", encoding='utf-8') as f:
            json.dump(reasoning_dict, f, ensure_ascii=False, indent=2)
        
        save_duration = time.time() - save_start
        # Print terminal warning with save duration
        print(f"[REWARD LOG] Reasoning dictionary saved to {log_file} in {save_duration:.4f} seconds. Total reasonings: {reasoning_dict['total_reasonings']}")
    except Exception as e:
        print(f"[REWARD LOG] Failed to save reasoning log: {e}")

    return rewards
