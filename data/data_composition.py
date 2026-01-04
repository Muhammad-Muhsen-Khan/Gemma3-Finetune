import json
from pathlib import Path

import numpy as np
import pandas as pd

from Helper.data_classes import PatientEntriesDataset, MD_ml_Dataset, MD_balanced_Dataset


def labels_vector_to_names(labels_vec: np.ndarray, idx_to_labels: dict[int, str]) -> list[str]:
    """Convert a multi-hot vector into a list of label names."""
    # labels_vec is float32 (0/1). Treat >0.5 as active.
    active = np.where(labels_vec > 0.5)[0].tolist()
    return [str(idx_to_labels.get(i, i)) for i in active]


def build_io_schemas_json() -> tuple[str, str]:
    """
    Build JSON schemas similar to the Pydantic schemas used in the provided script.
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
                        "entry": {"type": "string", "description": "patient entry"},
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
                        "entry_predictions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 1 or more medical symptoms present in the entry from the given medical symptoms list",
                        },
                    },
                    "required": ["entry_id", "entry_predictions"],
                },
            }
        },
        "required": ["predictions"],
    }
    return json.dumps(inputs_schema, indent=2), json.dumps(outputs_schema, indent=2)


def make_few_shot_examples(
    train_dataset: MD_balanced_Dataset,
    idx_to_labels: dict[int, str],
    k: int = 5,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Create few-shot sample input/output payloads (matching the user's script style).
    """
    rng = np.random.default_rng(seed)
    n = len(train_dataset)
    k = min(k, n)
    sample_indices = rng.choice(n, size=k, replace=False).tolist()

    entries = []
    predictions = []
    for entry_id, ds_idx in enumerate(sample_indices):
        text, y = train_dataset[ds_idx]
        label_names = labels_vector_to_names(np.asarray(y), idx_to_labels)
        entries.append({"entry_id": entry_id, "entry": str(text)})
        predictions.append({"entry_id": entry_id, "entry_predictions": label_names})

    few_shot_input = {"entries": entries}
    few_shot_output = {"predictions": predictions}
    return few_shot_input, few_shot_output


def make_system_prompt(
    symptoms_list: list[str],
    inputs_schema_json: str,
    outputs_schema_json: str,
    few_shot_input: dict,
    few_shot_output: dict,
) -> str:
    # This mirrors the prompt structure from the user's provided script.
    return f"""
        Extract the medical symptom(s) in the patient entries, selecting from the given list of all possible medical symptoms. Extract any predictions from the following medical list. 
        Do not include any symptoms outside of this list even if you think it is relevant.

        Medical symptoms: {symptoms_list}

        Respond with a single JSON object. Do not add any text before or after the JSON object. Follow the following JSON schemas:

        Input JSON schema: {inputs_schema_json}
        Output JSON schema: {outputs_schema_json}

        Sample input:
        {json.dumps(few_shot_input, indent=2)}

        Sample output:
        {json.dumps(few_shot_output, indent=2)}
    """.strip()


def make_sft_example(
    example_id: int,
    note_text: str,
    label_names: list[str],
    system_prompt: str,
) -> dict:
    """
    Create an SFT example in the expected LLaVA-style JSON format:
      { "id": "...", "conversations": [ {"from":"human","value":...}, {"from":"gpt","value":...} ] }
    """
    input_obj = {"entries": [{"entry_id": int(example_id), "entry": str(note_text).strip()}]}
    output_obj = {"predictions": [{"entry_id": int(example_id), "entry_predictions": label_names}]}

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
    # Load label mapping
    with open('/root/Gemma3-Finetune/data/Datasets/MD_sl/label_mapping.json', 'r') as file:
        label_mapping = json.load(file)

    # Test datasets
    data_path = ['/root/Gemma3-Finetune/data/Datasets/MD_sl/UpdatedSingletonTrainValEntries_102k.csv']
    dataset_wout_level_0 = PatientEntriesDataset(data_path, label_mapping=label_mapping)

    labels_to_idx = dataset_wout_level_0.labels_to_idx
    idx_to_labels = dataset_wout_level_0.idx_to_labels
    n_classes = len(labels_to_idx)

    MD_sl_test = PatientEntriesDataset(
        ['/root/Gemma3-Finetune/data/Datasets/MD_sl/SingletonTestEntries_10k.csv'], 
        label_mapping=label_mapping, 
        labels_to_idx=labels_to_idx)

    MD_ml_test = MD_ml_Dataset('/root/Gemma3-Finetune/data/Datasets/MD_ml/new_multi_label_test_data.csv', 
                                label_mapping, labels_to_idx=labels_to_idx, 
                                data_split="test", train_size=6000, test_size=2000, val_size=302)

    print(f"Test Datasets: \n\tMD_sl: {len(MD_sl_test)}\n\tMD_ml: {len(MD_ml_test)}")

    # Training datasets
    data_path = '/root/Gemma3-Finetune/data/Datasets/MD_ml/new_multi_label_test_data.csv'
    md_ml_train_dataset = MD_ml_Dataset(data_path, label_mapping, labels_to_idx=labels_to_idx, 
                                        data_split="train", train_size=6000, test_size=2000, val_size=302)
    md_ml_val_dataset = MD_ml_Dataset(data_path, label_mapping, labels_to_idx=labels_to_idx, 
                                        data_split="val", train_size=6000, test_size=2000, val_size=302)

    MD_ml_labels = np.stack([dataset[idx][1] for dataset in [md_ml_train_dataset, md_ml_val_dataset] 
                            for idx in range(len(dataset))])
    MD_ml_data = [dataset.data[idx][-1] for dataset in [md_ml_train_dataset, md_ml_val_dataset] 
                    for idx in range(len(dataset))]

    # Sample from MD_sl train & combine with MD_ml for balanced dataset
    MD_sl_dataset = pd.concat([dataset_wout_level_0.df, dataset_wout_level_0.L0])
    MD_sl_labels = MD_sl_dataset['Label'].to_numpy()
    MD_sl_data = MD_sl_dataset['Entry'].to_numpy().tolist()

    MD_sl_multi_label = np.zeros((len(MD_sl_labels), MD_ml_labels.shape[1]))
    for idx, label in enumerate(MD_sl_labels):
        MD_sl_multi_label[idx, label] = 1

    # Use Python list concatenation to avoid numpy dtype coercion for text
    all_data = list(MD_ml_data) + list(MD_sl_data)
    all_labels = np.vstack([MD_ml_labels, MD_sl_multi_label])
    # all_data = MD_sl_data
    # all_labels = MD_sl_multi_label

    # Create final train dataset (ONLY; this script is for generating training SFT data)
    train_dataset = MD_balanced_Dataset(
        '',
        texts=all_data,
        labels=all_labels,
        data_split="train",
        train_size=0.95,
    )

    print(f"Datasets:\n\tTrain: {len(train_dataset)}\n\tn_classes: {n_classes}")

    # ============================================================================
    # Convert training dataset to SFT JSON format and save
    # ============================================================================
    script_dir = Path(__file__).parent
    output_path = script_dir / "/root/Gemma3-Finetune/data/train_md_symptoms_sft.json"

    print("\nConverting training dataset to SFT JSON format...")
    sft_data: list[dict] = []

    symptoms_list = [idx_to_labels[i] for i in range(len(idx_to_labels))]
    inputs_schema_json, outputs_schema_json = build_io_schemas_json()
    few_shot_input, few_shot_output = make_few_shot_examples(train_dataset, idx_to_labels, k=5, seed=42)
    system_prompt = make_system_prompt(symptoms_list, inputs_schema_json, outputs_schema_json, few_shot_input, few_shot_output)

    for i in range(len(train_dataset)):
        text, y = train_dataset[i]
        label_names = labels_vector_to_names(np.asarray(y), idx_to_labels)
        sft_data.append(make_sft_example(i, text, label_names, system_prompt))

    with open(output_path, "w") as f:
        json.dump(sft_data, f, indent=2)

    print(f"Saved SFT training data to: {output_path}")