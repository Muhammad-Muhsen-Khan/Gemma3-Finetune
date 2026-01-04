import json
from pathlib import Path
import random
import torch
from pydantic import Field, BaseModel

from Helper.data_classes import MIMICIVDataset


class MedicalNoteInput(BaseModel):
    note_id: int 
    medical_note: str = Field(..., description="Full medical note text from MIMIC-IV dataset")

class ICD10CodeOutput(BaseModel):
    note_id: int 
    icd10_codes: list[str] = Field(..., description="List of generic ICD-10 CM codes (3-character codes) that apply to the medical note. Only include codes that are directly supported by the clinical documentation.")

class Inputs(BaseModel):
    notes: list[MedicalNoteInput] 

class Outputs(BaseModel):
    predictions: list[ICD10CodeOutput]


def labels_vector_to_names(labels_vec, idx_to_labels: dict[int, str]) -> list[str]:
    """Convert a multi-hot vector into a list of ICD-10 code names."""
    # Handle torch tensors
    if isinstance(labels_vec, torch.Tensor):
        active = (labels_vec == 1).nonzero(as_tuple=True)[0].tolist()
    else:
        # numpy array
        import numpy as np
        active = np.where(labels_vec > 0.5)[0].tolist()
    return [str(idx_to_labels.get(i, i)) for i in active]


def make_few_shot_examples(
    train_dataset,
    idx_to_labels: dict[int, str],
    k: int = 1,
    seed: int = 42,
) -> tuple[Inputs, Outputs]:
    """
    Create few-shot sample input/output payloads using Pydantic models.
    Uses only 1 example as specified.
    """
    random.seed(seed)
    n = len(train_dataset)
    k = min(k, n)
    sample_indices = random.sample(range(n), k)

    sample = []
    for idx, sample_idx in enumerate(sample_indices):
        text, labels_vector = train_dataset[sample_idx]
        codes = labels_vector_to_names(labels_vector, idx_to_labels)
        sample.append((idx, codes, text))

    few_shot_input = Inputs(
        notes=[
            MedicalNoteInput(note_id=id, medical_note=note_text.strip()) for (id, _, note_text) in sample
        ]
    )

    few_shot_output = Outputs(
        predictions=[
            ICD10CodeOutput(note_id=id, icd10_codes=codes) for (id, codes, _) in sample
        ]
    )

    return few_shot_input, few_shot_output


def make_system_prompt(
    inputs_schema: str,
    outputs_schema: str,
    few_shot_input: Inputs,
    few_shot_output: Outputs,
) -> str:
    """
    Create system prompt for ICD-10 CM code extraction from medical notes.
    """
    return f"""You are a medical coding expert. Your task is to extract all applicable generic ICD-10 CM codes (3-character codes) from medical notes.

Important guidelines:
1. Extract ONLY generic ICD-10 CM codes (3-character codes like "A00", "E11", "I50", etc.)
2. Do NOT include more specific codes (4+ character codes like "A00.0", "E11.9", etc.)
3. Only include codes that are directly supported by the clinical documentation in the note
4. A code should be included if the medical note contains sufficient clinical evidence for that diagnosis/condition
5. Return an empty list if no applicable codes are found
6. Codes must be valid ICD-10 CM format (3 uppercase letters/numbers, e.g., "A00", "Z99")

Respond with a single JSON object following the exact schema provided. Do not add any text before or after the JSON object.

Input JSON schema: {inputs_schema}
Output JSON schema: {outputs_schema}

Example input:
{few_shot_input.model_dump_json()}

Example output:
{few_shot_output.model_dump_json()}

""".strip()


def make_sft_example(
    example_id: int,
    note_text: str,
    icd10_codes: list[str],
    system_prompt: str,
) -> dict:
    """
    Create an SFT example in the expected LLaVA-style JSON format:
      { "id": "...", "conversations": [ {"from":"human","value":...}, {"from":"gpt","value":...} ] }
    """
    input_obj = Inputs(notes=[MedicalNoteInput(note_id=int(example_id), medical_note=str(note_text).strip())])
    output_obj = Outputs(predictions=[ICD10CodeOutput(note_id=int(example_id), icd10_codes=icd10_codes)])

    prompt = f"{system_prompt}\n\nInput: {input_obj.model_dump_json()}"
    answer = output_obj.model_dump_json()
    return {
        "id": f"{example_id:012d}",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
    }


if __name__ == "__main__":
    random.seed(42)

    # Training dataset - get labels_to_idx from train_dataset
    train_dataset = MIMICIVDataset(
        data_split="train"
    )

    # Get labels_to_idx from train_dataset BEFORE creating test_dataset
    labels_to_idx = train_dataset.labels_to_idx
    idx_to_labels = train_dataset.idx_to_labels
    n_classes = train_dataset.n_classes

    # Now create test_dataset with the same labels_to_idx
    test_dataset = MIMICIVDataset(
        data_split="test",
        labels_to_idx=labels_to_idx
    )

    print(f"Datasets:\n\tTrain: {len(train_dataset)}\n\tTest: {len(test_dataset)}\n\tn_classes: {n_classes}")

    # Generate JSON schema using Pydantic
    inputs_schema = json.dumps(Inputs.model_json_schema(), indent=2)
    outputs_schema = json.dumps(Outputs.model_json_schema(), indent=2)

    # Create few-shot examples using train_dataset (only 1 example)
    few_shot_input, few_shot_output = make_few_shot_examples(
        train_dataset, 
        idx_to_labels, 
        k=1, 
        seed=42
    )

    # Create system prompt
    system_prompt = make_system_prompt(
        inputs_schema,
        outputs_schema,
        few_shot_input,
        few_shot_output,
    )

    # ============================================================================
    # Convert training dataset to SFT JSON format and save
    # ============================================================================
    script_dir = Path(__file__).parent
    output_path = script_dir / "train_mimiciv_sft.json"

    print("\nConverting MIMIC-IV training dataset to SFT JSON format...")
    sft_data: list[dict] = []

    for i in range(len(train_dataset)):
        text, y = train_dataset[i]
        icd10_codes = labels_vector_to_names(y, idx_to_labels)
        sft_data.append(make_sft_example(i, text, icd10_codes, system_prompt))

    with open(output_path, "w") as f:
        json.dump(sft_data, f, indent=2)

    print(f"Saved MIMIC-IV SFT training data to: {output_path}")

