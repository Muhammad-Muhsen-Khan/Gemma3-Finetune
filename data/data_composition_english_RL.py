import json
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoProcessor


def make_system_prompt() -> str:
    """
    Create system prompt for SNOMED code and label prediction task.
    Uses reasoning format but WITHOUT few-shot examples (for RL training).
    """
    return """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.

You are a medical coding assistant. Given a patient clinical entry, identify the corresponding SNOMED CT concept code and clinical finding or disorder description.

Your task is to:
1. Analyze the patient entry text
2. Think through the reasoning process for identifying the medical finding or disorder
3. Provide the SNOMED CT concept ID (SCTID) and the corresponding SNOMED description

Now, analyze the following patient entry and provide your reasoning and answer in the required format.""".strip()


def make_rl_example(
    example_id: int,
    entry_text: str,
    sct_id: str,
    label: str,
    system_prompt: str,
    reasoning: str | None = None,
) -> dict:
    """
    Create an RL example in LLaVA-style JSON format for GRPO training.
    
    Args:
        example_id: Unique ID for the example
        entry_text: The patient clinical entry text
        sct_id: SNOMED CT concept ID
        label: SNOMED CT label
        system_prompt: The system prompt to use
        reasoning: Ignored for ground truth but kept for API compatibility
    
    Returns:
        Dictionary in LLaVA format with ground truth as a JSON dictionary string
    """
    entry_text = str(entry_text).strip()
    sct_id = str(sct_id).strip()
    label = str(label).strip()
    
    # Format user message: system prompt + patient entry
    user_message = f"{system_prompt}\n\nUser: {entry_text}.\nAssistant: "
    
    # Format ground truth as a JSON dictionary for the reward function
    # Reasoning is omitted as per user request
    ground_truth = {
        "id": f"{example_id:012d}",
        "sct_id": sct_id,
        "label": label
    }
    assistant_message = json.dumps(ground_truth, ensure_ascii=False)
        
        return {
            "id": f"{example_id:012d}",
            "conversations": [
            {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_message},
            ],
        }


if __name__ == "__main__":
    # Load the SNOMED findings dataset
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "datasets" / "snomed_synthesis_dataset_train.csv"
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    # Ensure required columns exist
    required_cols = ['entry', 'sct_id', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if reasoning column exists (optional, will use empty string if missing)
    has_reasoning = 'reasoning' in df.columns
    if has_reasoning:
        print("Found 'reasoning' column - will include reasoning in outputs")
    else:
        print("No 'reasoning' column found - will use empty reasoning")
    
    # Remove rows with missing values in required columns
    df = df.dropna(subset=required_cols)
    
    # Convert to string to ensure proper formatting
    df['sct_id'] = df['sct_id'].astype(str)
    df['label'] = df['label'].astype(str)
    df['entry'] = df['entry'].astype(str)
    
    if has_reasoning:
        df['reasoning'] = df['reasoning'].astype(str)
    
    print(f"Loaded {len(df)} entries")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Unique SNOMED codes: {df['sct_id'].nunique()}")
    
    # ============================================================================
    # Convert dataset to RL JSON format (LLaVA format for GRPO)
    # ============================================================================
    output_path = script_dir / "train_snomed_prediction_rl.json"
    
    print("\nConverting dataset to RL JSON format...")
    rl_data: list[dict] = []
    
    # Create system prompt (no few-shot examples for RL)
    system_prompt = make_system_prompt()
    
    for idx, row in df.iterrows():
        # Get reasoning if available
        reasoning_value = None
        if has_reasoning:
            if pd.notna(row.get('reasoning')) and str(row.get('reasoning')).strip():
                reasoning_value = str(row['reasoning']).strip()
        
        example = make_rl_example(
            idx,
            row['entry'],
            row['sct_id'],
            row['label'],
            system_prompt,
            reasoning=reasoning_value,
        )
        rl_data.append(example)
        
    # Save the dataset
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved RL training data to: {output_path}")
    print(f"Total examples: {len(rl_data)}")
    
    # Print a random example to verify format
    if rl_data:
        rng = np.random.default_rng(42)
        random_idx = rng.choice(len(rl_data))
        random_example = rl_data[random_idx]
        
        print(f"\n{'='*80}")
        print(f"Random Example (index {random_idx}):")
        print(f"{'='*80}")
        print(f"\nPROMPT:")
        print(f"{'-'*80}")
        print(random_example["conversations"][0]["value"])
        print(f"\n{'='*80}")
        print(f"GROUND TRUTH OUTPUT:")
        print(f"{'-'*80}")
        print(random_example["conversations"][1]["value"])
        print(f"{'='*80}\n")
        
        # Demonstrate parsing for reward functions
        ground_truth_str = random_example["conversations"][1]["value"]
        try:
            gt_data = json.loads(ground_truth_str)
            print(f"Parsed ground truth (for reward function):")
            print(f"  ID: {gt_data.get('id')}")
            print(f"  SCTID: {gt_data.get('sct_id')}")
            print(f"  Label: {gt_data.get('label')}")
        except json.JSONDecodeError:
            print("Error: Ground truth is not valid JSON")
        print(f"{'='*80}\n")
