import os
import re
import json
from datetime import datetime
# from math_verify import parse, verify  # Commented out - not needed for JSON-based rewards

# ============================================================================
# ORIGINAL IMPLEMENTATIONS (COMMENTED OUT - PRESERVED FOR REFERENCE)
# ============================================================================

# def accuracy_reward(completions, assistant, **kwargs):
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

# ============================================================================
# NEW IMPLEMENTATIONS FOR JSON/SNOMED PREDICTION TASK
# ============================================================================

def extract_json(text: str) -> dict:
    """
    Extract JSON object from text that might have extra content before/after.
    Handles cases where the model outputs text before or after the JSON.
    """
    # Try to find JSON object in the text by looking for {...} pattern
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found complete JSON object
                json_str = text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to find next JSON object
                    start_idx = -1
                    continue
    
    # If no JSON found via brace matching, try parsing the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def accuracy_reward(completions, assistant, **kwargs):
    """
    Reward function for SNOMED prediction task - checks JSON accuracy.
    Compares the sct_id and label from the model's prediction against the ground truth.
    
    Returns:
        - 1.0 if both sct_id and label match
        - 0.5 if only sct_id matches (partial credit)
        - 0.0 otherwise
    """
    contents = [completion[0]["content"] for completion in completions]
    solution = [a['content'] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        try:
            # Parse both completion and solution as JSON
            content_json = extract_json(content)
            sol_json = json.loads(sol.strip())  # Solution should be clean JSON
            
            if content_json and sol_json:
                # Compare predictions
                pred_completion = content_json.get("predictions", [])
                pred_solution = sol_json.get("predictions", [])
                
                if len(pred_completion) > 0 and len(pred_solution) > 0:
                    comp_pred = pred_completion[0]
                    sol_pred = pred_solution[0]
                    
                    # Check if sct_id and label match (case-insensitive for labels)
                    comp_sct_id = comp_pred.get("sct_id", "").strip()
                    sol_sct_id = sol_pred.get("sct_id", "").strip()
                    comp_label = comp_pred.get("label", "").strip()
                    sol_label = sol_pred.get("label", "").strip()
                    
                    sct_match = comp_sct_id == sol_sct_id
                    label_match = comp_label.lower() == sol_label.lower()
                    
                    # Reward = 1.0 if both match, 0.5 if only sct_id matches, 0.0 otherwise
                    if sct_match and label_match:
                        reward = 1.0
                    elif sct_match:
                        reward = 0.5  # Partial credit for correct code
                    else:
                        reward = 0.0
                        
        except Exception as e:
            # If parsing fails, reward is 0.0
            reward = 0.0
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward error: {e} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    
    return rewards


def format_reward(completions, **kwargs):
    """
    Reward function that checks if the completion has valid JSON structure.
    Validates that the output is parseable JSON with the expected structure:
    {"predictions": [{"sct_id": "...", "label": "..."}]}
    
    Returns:
        - 1.0 if valid JSON with correct structure
        - 0.0 if invalid JSON or missing required fields
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        reward = 0.0
        
        try:
            # Try to extract and parse JSON
            content_json = extract_json(content)
            
            if content_json is None:
                reward = 0.0
            else:
                # Check if it has the expected structure
                if "predictions" in content_json:
                    predictions = content_json["predictions"]
                    if isinstance(predictions, list) and len(predictions) > 0:
                        first_pred = predictions[0]
                        # Check if it has required fields
                        if isinstance(first_pred, dict) and "sct_id" in first_pred and "label" in first_pred:
                            reward = 1.0
                        else:
                            reward = 0.0
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
                    
        except Exception:
            # If any error occurs, reward is 0.0
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

