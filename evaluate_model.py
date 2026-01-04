import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re
import sys

# Add paths for imports
sys.path.append('data')
sys.path.append('src')

# Import data classes
from Helper.data_classes import MD_ml_Dataset, PatientEntriesDataset
from data_composition import (
    build_io_schemas_json, 
    make_few_shot_examples, 
    make_system_prompt,
    labels_vector_to_names
)

def load_model_and_processor(checkpoint_path, device="cuda"):
    """Load the trained model and processor from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    
    return model, processor

def extract_json_from_text(text):
    """Extract JSON object from model output, handling potential extra text."""
    # Try to find JSON object in the text
    # Look for { ... } pattern
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # If that fails, try to parse the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def predict_symptoms(model, processor, text, system_prompt, device="cuda"):
    """Run inference on a single text sample."""
    # Format input the same way as training
    input_obj = {"entries": [{"entry_id": 0, "entry": str(text).strip()}]}
    prompt = f"{system_prompt}\n\nInput: {json.dumps(input_obj, ensure_ascii=False)}"
    
    # Format as conversation
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
    
    # Decode
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the model's response (after the user prompt)
    # The model should output JSON
    response = generated_text.split(prompt)[-1].strip()
    
    # Parse JSON
    result = extract_json_from_text(response)
    
    if result and "predictions" in result and len(result["predictions"]) > 0:
        predicted_symptoms = result["predictions"][0].get("entry_predictions", [])
        return predicted_symptoms
    else:
        return []

def evaluate_model(checkpoint_path, test_dataset, idx_to_labels, device="cuda"):
    """Evaluate the model on test dataset."""
    
    # Load model
    model, processor = load_model_and_processor(checkpoint_path, device)
    
    # Build system prompt (same as training)
    symptoms_list = [idx_to_labels[i] for i in range(len(idx_to_labels))]
    inputs_schema_json, outputs_schema_json = build_io_schemas_json()
    
    # Use a small subset for few-shot examples (or load from training data)
    # For now, using first 5 examples
    few_shot_dataset = MD_ml_Dataset(
        'Datasets/MD_ml/new_multi_label_test_data.csv',
        test_dataset.labels_to_idx,
        labels_to_idx=test_dataset.labels_to_idx,
        data_split="train",
        train_size=6000,
        test_size=2000,
        val_size=302
    )
    few_shot_input, few_shot_output = make_few_shot_examples(
        few_shot_dataset, idx_to_labels, k=5, seed=42
    )
    
    system_prompt = make_system_prompt(
        symptoms_list, inputs_schema_json, outputs_schema_json,
        few_shot_input, few_shot_output
    )
    
    # Run evaluation
    all_predictions = []
    all_labels = []
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    for idx in tqdm(range(len(test_dataset))):
        text, labels_vector = test_dataset[idx]
        
        # Get ground truth labels
        gt_labels = labels_vector_to_names(labels_vector.numpy(), idx_to_labels)
        
        # Predict
        try:
            pred_labels = predict_symptoms(model, processor, text, system_prompt, device)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            pred_labels = []
        
        all_predictions.append(pred_labels)
        all_labels.append(gt_labels)
    
    # Calculate metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # Convert to binary vectors for metric calculation
    n_classes = len(idx_to_labels)
    y_true = np.zeros((len(all_labels), n_classes))
    y_pred = np.zeros((len(all_predictions), n_classes))
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    
    for i, (gt, pred) in enumerate(zip(all_labels, all_predictions)):
        for label in gt:
            if label in labels_to_idx:
                y_true[i, labels_to_idx[label]] = 1
        for label in pred:
            if label in labels_to_idx:
                y_pred[i, labels_to_idx[label]] = 1
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Calculate exact match accuracy
    exact_match = np.all(y_true == y_pred, axis=1).mean()
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Micro Precision: {precision:.4f}")
    print(f"Micro Recall: {recall:.4f}")
    print(f"Micro F1: {f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Exact Match Accuracy: {exact_match:.4f}")
    print("="*50)
    
    return {
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'exact_match': exact_match,
        'predictions': all_predictions,
        'ground_truth': all_labels
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint directory (e.g., output/md_symptoms/checkpoint-5)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load label mapping
    with open('Datasets/MD_sl/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    # Load test dataset
    # First, we need to get labels_to_idx from a training dataset
    train_dataset = PatientEntriesDataset(
        ['Datasets/MD_sl/UpdatedSingletonTrainValEntries_102k.csv'],
        label_mapping=label_mapping
    )
    labels_to_idx = train_dataset.labels_to_idx
    idx_to_labels = train_dataset.idx_to_labels
    
    # Load test dataset
    test_dataset = MD_ml_Dataset(
        'Datasets/MD_ml/new_multi_label_test_data.csv',
        label_mapping,
        labels_to_idx=labels_to_idx,
        data_split="test",
        train_size=6000,
        test_size=2000,
        val_size=302
    )
    
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Run evaluation
    results = evaluate_model(args.checkpoint_path, test_dataset, idx_to_labels, args.device)
    
    # Save results
    # Convert numpy arrays to lists for JSON serialization
    results_save = {
        'metrics': {k: float(v) for k, v in results.items() if k not in ['predictions', 'ground_truth']},
        'predictions': results['predictions'],
        'ground_truth': results['ground_truth']
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results_save, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")

