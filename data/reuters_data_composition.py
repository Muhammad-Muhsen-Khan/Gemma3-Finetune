import json
from pathlib import Path
import random
import torch
from pydantic import Field, BaseModel

from Helper.data_classes import reutersDataset


class EntryInput(BaseModel):
    entry_id: int 
    entry: str = Field(..., description="Reuters Entry")

class EntryOutput(BaseModel):
    entry_id: int 
    entry_predictions: list[str] = Field(..., description="List of 1 or more Reuters labels present in the entry from the given Reuters labels")

class Inputs(BaseModel):
    entries: list[EntryInput] 

class Outputs(BaseModel):
    predictions: list[EntryOutput]


def labels_vector_to_names(labels_vec, idx_to_labels: dict[int, str]) -> list[str]:
    """Convert a multi-hot vector into a list of label names."""
    # Handle torch tensors
    if isinstance(labels_vec, torch.Tensor):
        active = (labels_vec == 1).nonzero(as_tuple=True)[0].tolist()
    else:
        # numpy array
        import numpy as np
        active = np.where(labels_vec > 0.5)[0].tolist()
    return [str(idx_to_labels.get(i, i)) for i in active]


def make_few_shot_examples(
    reuters_ml_train_dataset,
    idx_to_labels: dict[int, str],
    k: int = 5,
    seed: int = 42,
) -> tuple[Inputs, Outputs]:
    """
    Create few-shot sample input/output payloads using Pydantic models.
    Uses Reuters_ml_train for few-shot examples as in the provided script.
    """
    random.seed(seed)
    n = len(reuters_ml_train_dataset)
    k = min(k, n)
    sample_indices = random.sample(range(n), k)

    unformatted_sample = [reuters_ml_train_dataset[i] for i in sample_indices]
    sample = []

    for idx, (entry, labels_vector) in enumerate(unformatted_sample):
        labels = [idx_to_labels[lbl] for lbl in (labels_vector == 1).nonzero(as_tuple=True)[0].tolist()]
        sample.append((idx, labels, entry))

    few_shot_input = Inputs(
        entries=[
            EntryInput(entry_id=id, entry=entry) for (id, _, entry) in sample
        ]
    )

    few_shot_output = Outputs(
        predictions=[
            EntryOutput(entry_id=id, entry_predictions=lbls) for (id, lbls, _) in sample
        ]
    )

    return few_shot_input, few_shot_output


def make_system_prompt(
    labels_list: list[str],
    inputs_schema: str,
    outputs_schema: str,
    few_shot_input: Inputs,
    few_shot_output: Outputs,
) -> str:
    """
    Create system prompt using the exact format from the provided script.
    """
    return f"""
        Extract the correct label(s) for the Reuters news entry, selecting from the given list of all possible Reuters labels. Extract any predictions from the following label list. 
        Do not include any labels outside of this list even if you think it is relevant.

        Reuters labels: {labels_list}

        Respond with a single JSON object. Do not add any text before or after the JSON object. Follow the following JSON schemas:

        Input JSON schema: {inputs_schema}
        Output JSON schema: {outputs_schema}

        Sample input:
        {few_shot_input.model_dump_json()}

        Sample output:
        {few_shot_output.model_dump_json()}

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
    input_obj = Inputs(entries=[EntryInput(entry_id=int(example_id), entry=str(note_text).strip())])
    output_obj = Outputs(predictions=[EntryOutput(entry_id=int(example_id), entry_predictions=label_names)])

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

    # Test datasets
    Reuters_sl_test = reutersDataset(
        data_split="single_test",
        single_or_multi_labels="multi"
    )

    Reuters_ml_test = reutersDataset(
        data_split="multi_test",
        single_or_multi_labels="multi"
    )

    labels_to_idx = Reuters_sl_test.labels_to_idx
    idx_to_labels = Reuters_sl_test.idx_to_labels
    labels_list = list(labels_to_idx.keys())

    print(f"Test Datasets: \n\tReuters_sl: {len(Reuters_sl_test)}\n\tReuters_ml: {len(Reuters_ml_test)}")

    # Training and Validation datasets
    train_dataset = reutersDataset(
        data_split="single_train",
        single_or_multi_labels="multi"
    )

    val_dataset = reutersDataset(
        data_split="single_val",
        single_or_multi_labels="multi"
    )

    # Reuters_ml_train for few-shot examples (as in the provided script)
    Reuters_ml_train = reutersDataset(
        data_split="multi_train",
        single_or_multi_labels="multi"
    )

    n_classes = train_dataset.n_classes

    print(f"Datasets:\n\tTrain: {len(train_dataset)}\n\tVal: {len(val_dataset)}\n\tn_classes: {n_classes}")

    # Generate JSON schema using Pydantic
    inputs_schema = json.dumps(Inputs.model_json_schema(), indent=2)
    outputs_schema = json.dumps(Outputs.model_json_schema(), indent=2)

    # Create few-shot examples using Reuters_ml_train
    few_shot_input, few_shot_output = make_few_shot_examples(
        Reuters_ml_train, 
        idx_to_labels, 
        k=5, 
        seed=42
    )

    # Create system prompt
    system_prompt = make_system_prompt(
        labels_list,
        inputs_schema,
        outputs_schema,
        few_shot_input,
        few_shot_output,
    )

    # ============================================================================
    # Convert training dataset to SFT JSON format and save
    # ============================================================================
    script_dir = Path(__file__).parent
    output_path = script_dir / "train_reuters_sft.json"

    print("\nConverting Reuters training dataset to SFT JSON format...")
    sft_data: list[dict] = []

    for i in range(len(train_dataset)):
        text, y = train_dataset[i]
        label_names = labels_vector_to_names(y, idx_to_labels)
        sft_data.append(make_sft_example(i, text, label_names, system_prompt))

    with open(output_path, "w") as f:
        json.dump(sft_data, f, indent=2)

    print(f"Saved Reuters SFT training data to: {output_path}")

