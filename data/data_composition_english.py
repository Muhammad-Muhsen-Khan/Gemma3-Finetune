import json
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoProcessor


def format_few_shot_example(entry: str, sct_id: str, label: str, reasoning: str | None = None, include_reasoning: bool = False) -> str:
    """
    Format a few-shot example in plain English.
    Note: Few-shot examples always use the simple format without reasoning tags,
    even when reasoning is enabled (reasoning tags are only used in actual SFT examples).
    
    Args:
        entry: Patient clinical entry text
        sct_id: SNOMED CT concept ID
        label: SNOMED CT label
        reasoning: Optional reasoning text (not used in few-shot examples)
        include_reasoning: Whether reasoning is enabled (not used here, kept for API consistency)
    """
    # Always use simple format for few-shot examples
    example = f"Patient entry: {entry}\n"
    example += f"SCTID: {sct_id}, SNOMED description: {label}."
    return example


def make_few_shot_examples(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
    include_reasoning: bool = False,
) -> str:
    """
    Create few-shot examples in plain English format for SNOMED prediction task.
    
    Args:
        df: DataFrame with the data
        k: Number of examples to sample
        seed: Random seed
        include_reasoning: If True, include reasoning in examples
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    k = min(k, n)
    sample_indices = rng.choice(n, size=k, replace=False).tolist()

    examples = []
    for idx in sample_indices:
        row = df.iloc[idx]
        entry = str(row['entry']).strip()
        sct_id = str(row['sct_id']).strip()
        label = str(row['label']).strip()
        
        reasoning = None
        if include_reasoning and 'reasoning' in row and pd.notna(row['reasoning']):
            reasoning = str(row['reasoning']).strip()
        
        example = format_few_shot_example(entry, sct_id, label, reasoning, include_reasoning=include_reasoning)
        examples.append(example)
    
    return "\n\n".join(examples)


def make_system_prompt(
    few_shot_examples: str,
    include_reasoning: bool = False,
) -> str:
    """
    Create system prompt for SNOMED code and label prediction task in plain English.
    
    Args:
        few_shot_examples: Few-shot examples string
        include_reasoning: If True, use the reasoning format with tags
    """
    if include_reasoning:
        return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.

You are a medical coding assistant. Given a patient clinical entry, identify the corresponding SNOMED CT concept code and clinical finding or disorder description.

Your task is to:
1. Analyze the patient entry text
2. Think through the reasoning process for identifying the medical finding or disorder
3. Provide the SNOMED CT concept ID (SCTID) and the corresponding SNOMED description

Here are some examples:

{few_shot_examples}

Now, analyze the following patient entry and provide your reasoning and answer in the required format.""".strip()
    else:
        return f"""
You are a medical coding assistant. Given a patient clinical entry, identify the corresponding SNOMED CT concept code and clinical finding or disorder description.

Your task is to:
1. Analyze the patient entry text
2. Identify the medical finding or disorder mentioned
3. Provide the SNOMED CT concept ID (SCTID) and the corresponding SNOMED description

Here are some examples:

{few_shot_examples}

Now, analyze the following patient entry and provide the SCTID and SNOMED description in a similar format.
    """.strip()


def make_sft_example(
    example_id: int,
    entry_text: str,
    sct_id: str,
    label: str,
    system_prompt: str,
    reasoning: str | None = None,
    include_reasoning: bool = False,
) -> dict:
    """
    Create an SFT example in the expected LLaVA-style JSON format:
      {{ "id": "...", "conversations": [ {{"from":"human","value":...}}, {{"from":"gpt","value":...}} ] }}
    
    Args:
        example_id: Unique ID for the example
        entry_text: The patient clinical entry text
        sct_id: SNOMED CT concept ID
        label: SNOMED CT label
        system_prompt: The system prompt to use
        reasoning: Optional reasoning text (if None, not included in output)
        include_reasoning: If True, format with reasoning tags
    """
    entry_text = str(entry_text).strip()
    sct_id = str(sct_id).strip()
    label = str(label).strip()
    
    if include_reasoning and reasoning is not None and str(reasoning).strip():
        # Format with reasoning tags: system prompt + user message + assistant response with tags
        user_message = f"User: {entry_text}"
        answer_text = f"SCTID: {sct_id}, SNOMED description: {label}."
        assistant_message = f"Assistant: <think> {reasoning} </think> <answer> {answer_text} </answer>"
        
        return {
            "id": f"{example_id:012d}",
            "conversations": [
                {"from": "human", "value": f"{system_prompt}\n\n{user_message}"},
                {"from": "gpt", "value": assistant_message},
            ],
        }
    else:
        # Original format: system prompt with patient entry, then answer
        prompt = f"{system_prompt}\n\nPatient entry: {entry_text}"
        answer = f"SCTID: {sct_id}, SNOMED description: {label}."
        
        return {
            "id": f"{example_id:012d}",
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
        }


if __name__ == "__main__":
    # Flag to control whether to include reasoning in outputs
    include_reasoning = True
    
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
    
    # Check if reasoning column exists when flag is enabled
    if include_reasoning:
        if 'reasoning' not in df.columns:
            raise ValueError("include_reasoning is True but 'reasoning' column not found in dataset")
        print("Including reasoning in outputs")
    
    # Remove rows with missing values in required columns
    df = df.dropna(subset=required_cols)
    
    # Convert sct_id to string to ensure proper formatting
    df['sct_id'] = df['sct_id'].astype(str)
    df['label'] = df['label'].astype(str)
    df['entry'] = df['entry'].astype(str)
    
    # Convert reasoning to string if flag is enabled
    if include_reasoning:
        df['reasoning'] = df['reasoning'].astype(str)
    
    print(f"Loaded {len(df)} entries")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Unique SNOMED codes: {df['sct_id'].nunique()}")
    
    # ============================================================================
    # Convert dataset to SFT JSON format and save
    # ============================================================================
    output_path = script_dir / "train_snomed_prediction_sft.json"
    
    # Option to enable token counting (set to False to skip model loading)
    count_tokens = False
    
    print("\nConverting dataset to SFT JSON format...")
    sft_data: list[dict] = []
    
    # Load tokenizer for counting tokens (optional)
    tokenizer = None
    if count_tokens:
        print("Loading tokenizer...")
        model_id = "google/gemma-3-4b-it"
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
    
    few_shot_examples = make_few_shot_examples(df, k=5, seed=42, include_reasoning=include_reasoning)
    system_prompt = make_system_prompt(few_shot_examples, include_reasoning=include_reasoning)
    
    # Track maximum token counts (only if counting tokens)
    max_prompt_tokens = 0
    max_response_tokens = 0
    max_total_tokens = 0
    
    for idx, row in df.iterrows():
        if include_reasoning:
            if 'reasoning' in row and pd.notna(row.get('reasoning')) and str(row.get('reasoning')).strip():
                reasoning_value = str(row['reasoning']).strip()
            else:
                print(f"⚠️  Warning: Row {idx} (SCTID: {row['sct_id']}) is missing reasoning")
                reasoning_value = None
        else:
            reasoning_value = None
        
        example = make_sft_example(
            idx,
            row['entry'],
            row['sct_id'],
            row['label'],
            system_prompt,
            reasoning=reasoning_value,
            include_reasoning=include_reasoning
        )
        sft_data.append(example)
        
        # Tokenize prompt and response (only if tokenizer is loaded)
        if tokenizer is not None:
            prompt_text = example["conversations"][0]["value"]
            response_text = example["conversations"][1]["value"]
            
            prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            response_tokens = len(tokenizer.encode(response_text, add_special_tokens=False))
            total_tokens = prompt_tokens + response_tokens
            
            # Update maximums
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
            max_response_tokens = max(max_response_tokens, response_tokens)
            max_total_tokens = max(max_total_tokens, total_tokens)
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved SFT training data to: {output_path}")
    print(f"Total examples: {len(sft_data)}")
    
    if tokenizer is not None:
        print(f"\nToken Statistics:")
        print(f"  Maximum prompt tokens: {max_prompt_tokens}")
        print(f"  Maximum response tokens: {max_response_tokens}")
        print(f"  Maximum total tokens (prompt + response): {max_total_tokens}")
    
    # Print a random example
    if sft_data:
        rng = np.random.default_rng(42)
        random_idx = rng.choice(len(sft_data))
        random_example = sft_data[random_idx]
        
        print(f"\n{'='*80}")
        print(f"Random Example (index {random_idx}):")
        print(f"{'='*80}")
        print(f"\nPROMPT:")
        print(f"{'-'*80}")
        print(random_example["conversations"][0]["value"])
        print(f"\n{'='*80}")
        print(f"OUTPUT:")
        print(f"{'-'*80}")
        print(random_example["conversations"][1]["value"])
        print(f"{'='*80}\n")