import json
from pathlib import Path
import numpy as np
import pandas as pd


def build_io_schemas_json() -> tuple[str, str]:
    """
    Build JSON schemas for SNOMED code and label prediction task.
    Returned as (inputs_schema_json, outputs_schema_json) strings.
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
                        "entry_id": {"type": "integer"},
                        "entry": {"type": "string", "description": "patient clinical entry"},
                    },
                    "required": ["entry_id", "entry"],
                },
            }
        },
        "required": ["entries"],
    }
    outputs_schema = {
        "title": "Outputs",
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "title": "EntryOutput",
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "integer"},
                        "sct_id": {"type": "string", "description": "SNOMED CT concept ID"},
                        "label": {"type": "string", "description": "SNOMED CT clinical finding or disorder label"},
                    },
                    "required": ["entry_id", "sct_id", "label"],
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
) -> tuple[dict, dict]:
    """
    Create few-shot sample input/output payloads for SNOMED prediction task.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    k = min(k, n)
    sample_indices = rng.choice(n, size=k, replace=False).tolist()

    entries = []
    predictions = []
    for entry_id, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        entries.append({
            "entry_id": entry_id,
            "entry": str(row['entry']).strip()
        })
        predictions.append({
            "entry_id": entry_id,
            "sct_id": str(row['sct_id']).strip(),
            "label": str(row['label']).strip()
        })

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
) -> dict:
    """
    Create an SFT example in the expected LLaVA-style JSON format:
      {{ "id": "...", "conversations": [ {{"from":"human","value":...}}, {{"from":"gpt","value":...}} ] }}
    """
    input_obj = {
        "entries": [{
            "entry_id": int(example_id),
            "entry": str(entry_text).strip()
        }]
    }
    output_obj = {
        "predictions": [{
            "entry_id": int(example_id),
            "sct_id": str(sct_id).strip(),
            "label": str(label).strip()
        }]
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
    dataset_path = script_dir / "datasets" / "sampled_snomed_findings_2000_labels_closest_to_60.csv"
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
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
    
    print(f"Loaded {len(df)} entries")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Unique SNOMED codes: {df['sct_id'].nunique()}")
    
    # ============================================================================
    # Convert dataset to SFT JSON format and save
    # ============================================================================
    output_path = script_dir / "train_snomed_prediction_sft.json"
    
    print("\nConverting dataset to SFT JSON format...")
    sft_data: list[dict] = []
    
    inputs_schema_json, outputs_schema_json = build_io_schemas_json()
    few_shot_input, few_shot_output = make_few_shot_examples(df, k=5, seed=42)
    system_prompt = make_system_prompt(inputs_schema_json, outputs_schema_json, few_shot_input, few_shot_output)
    
    for idx, row in df.iterrows():
        sft_data.append(make_sft_example(
            idx,
            row['entry'],
            row['sct_id'],
            row['label'],
            system_prompt
        ))
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved SFT training data to: {output_path}")
    print(f"Total examples: {len(sft_data)}")