import json
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoProcessor


def build_io_schemas_json(include_reasoning: bool = False) -> tuple[str, str]:
    """
    Build JSON schemas for SNOMED code and label prediction task.
    Returned as (inputs_schema_json, outputs_schema_json) strings.
    
    Args:
        include_reasoning: If True, include reasoning property in outputs schema
    """
    inputs_schema = {
        "title": "Inputs",
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "title": "EntryInput",
                    "type": "object",
                    "properties": {
                        "entry": {"type": "string", "description": "patient clinical entry"},
                    },
                    "required": ["entry"],
                },
            }
        },
        "required": ["entries"],
    }
    
    # Build outputs schema properties
    output_properties = {}
    output_required = []
    
    if include_reasoning:
        output_properties["reasoning"] = {"type": "string", "description": "Reasoning for the SNOMED CT code and label selection"}
        output_required.append("reasoning")
    
    output_properties["sct_id"] = {"type": "string", "description": "SNOMED CT concept ID"}
    output_properties["label"] = {"type": "string", "description": "SNOMED CT clinical finding or disorder label"}
    output_required.extend(["sct_id", "label"])
    
    outputs_schema = {
        "title": "Outputs",
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "title": "EntryOutput",
                    "type": "object",
                    "properties": output_properties,
                    "required": output_required,
                },
            }
        },
        "required": ["predictions"],
    }
    return json.dumps(inputs_schema, indent=2), json.dumps(outputs_schema, indent=2)


def make_few_shot_examples(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
    include_reasoning: bool = False,
) -> tuple[dict, dict]:
    """
    Create few-shot sample input/output payloads for SNOMED prediction task.
    
    Args:
        df: DataFrame with the data
        k: Number of examples to sample
        seed: Random seed
        include_reasoning: If True, include reasoning in predictions
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    k = min(k, n)
    sample_indices = rng.choice(n, size=k, replace=False).tolist()

    entries = []
    predictions = []
    for idx in sample_indices:
        row = df.iloc[idx]
        entries.append({
            "entry": str(row['entry']).strip()
        })
        
        prediction = {}
        if include_reasoning and 'reasoning' in row and pd.notna(row['reasoning']):
            prediction["reasoning"] = str(row['reasoning']).strip()
        prediction["sct_id"] = str(row['sct_id']).strip()
        prediction["label"] = str(row['label']).strip()
        predictions.append(prediction)

    few_shot_input = {"entries": entries}
    few_shot_output = {"predictions": predictions}
    return few_shot_input, few_shot_output


def make_system_prompt(
    inputs_schema_json: str,
    outputs_schema_json: str,
    few_shot_input: dict,
    few_shot_output: dict,
) -> str:
    """
    Create system prompt for SNOMED code and label prediction task.
    """
    return f"""
You are a medical coding assistant. Given a patient clinical entry, identify the corresponding SNOMED CT concept code (sct_id) and clinical finding or disorder label.

Your task is to:
1. Analyze the patient entry text
2. Identify the medical finding or disorder mentioned
3. Return the correct SNOMED CT concept ID (sct_id) and the corresponding label name

The label should be a SNOMED CT clinical finding or disorder term, typically ending with "(finding)" or "(disorder)".

Respond with a single JSON object. Do not add any text before or after the JSON object. Follow the following JSON schemas:

Input JSON schema:
{inputs_schema_json}

Output JSON schema:
{outputs_schema_json}

Sample input:
{json.dumps(few_shot_input, indent=2, ensure_ascii=False)}

Sample output:
{json.dumps(few_shot_output, indent=2, ensure_ascii=False)}
    """.strip()


def make_sft_example(
    example_id: int,
    entry_text: str,
    sct_id: str,
    label: str,
    system_prompt: str,
    reasoning: str | None = None,
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
    """
    input_obj = {
        "entries": [{
            "entry": str(entry_text).strip()
        }]
    }
    
    prediction = {}
    if reasoning is not None and str(reasoning).strip():
        prediction["reasoning"] = str(reasoning).strip()
    prediction["sct_id"] = str(sct_id).strip()
    prediction["label"] = str(label).strip()
    
    output_obj = {
        "predictions": [prediction]
    }

    prompt = f"{system_prompt}\n\nInput: {json.dumps(input_obj, ensure_ascii=False)}"
    answer = json.dumps(output_obj, ensure_ascii=False)
    return {
        "id": f"{example_id:012d}",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
    }


if __name__ == "__main__":
    # Load the SNOMED findings dataset
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "datasets" / "snomed_synthesis_dataset.csv"
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    # Check if reasoning column exists
    has_reasoning = 'reasoning' in df.columns
    if has_reasoning:
        print("Found 'reasoning' column - will include in outputs schema")
    
    # Ensure required columns exist
    required_cols = ['entry', 'sct_id', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing values in required columns
    df = df.dropna(subset=required_cols)
    
    # Convert sct_id to string to ensure proper formatting
    df['sct_id'] = df['sct_id'].astype(str)
    df['label'] = df['label'].astype(str)
    df['entry'] = df['entry'].astype(str)
    
    # Convert reasoning to string if it exists
    if has_reasoning:
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
    
    inputs_schema_json, outputs_schema_json = build_io_schemas_json(include_reasoning=has_reasoning)
    few_shot_input, few_shot_output = make_few_shot_examples(df, k=5, seed=42, include_reasoning=has_reasoning)
    system_prompt = make_system_prompt(inputs_schema_json, outputs_schema_json, few_shot_input, few_shot_output)
    
    # Track maximum token counts (only if counting tokens)
    max_prompt_tokens = 0
    max_response_tokens = 0
    max_total_tokens = 0
    
    for idx, row in df.iterrows():
        if has_reasoning:
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
            reasoning=reasoning_value
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