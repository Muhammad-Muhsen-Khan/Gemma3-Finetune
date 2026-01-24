import os
import re
import json
import torch
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import GenerationConfig
from tqdm import tqdm
from src.constants import SYSTEM_MESSAGE, DEFAULT_END_TOKEN


class SNOMEDDataset(Dataset):
    """Dataset for SNOMED evaluation"""
    
    def __init__(self, csv_path: str, max_samples: Optional[int] = None):
        df = pd.read_csv(csv_path)
        if max_samples is not None:
            df = df.head(max_samples)
        self.sct_ids = df['sct_id'].astype(str).tolist()
        self.labels = df['label'].astype(str).tolist()
        self.entries = df['entry'].astype(str).tolist()
    
    def __len__(self):
        return len(self.sct_ids)
    
    def __getitem__(self, idx):
        return {
            'sct_id': self.sct_ids[idx],
            'label': self.labels[idx],
            'entry': self.entries[idx]
        }


def extract_snomed_codes(text):
    """
    Extract all SNOMED CT codes from a given piece of text.
    SNOMED CT codes are typically numeric strings (6-18 digits).
    Returns a list of unique SNOMED codes found in the text.
    Copied from reward_funcs.py to match reward function logic.
    """
    # First, try to extract from <answer> tags if present
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    search_text = answer_match.group(1).strip() if answer_match else text
    
    # Find all numeric sequences that could be SNOMED codes
    # SNOMED CT codes are typically 6-18 digits long
    # Pattern: sequences of digits, potentially with commas/spaces around them
    snomed_pattern = r'\b\d{6,18}\b'
    matches = re.findall(snomed_pattern, search_text)
    
    # Return unique codes (as strings)
    return list(set(matches))


def process_snomed_output(completion: str, gt_sct_id: str, gt_label: str) -> dict:
    """
    Process a single model output using reward function logic.
    - Extracts SNOMED codes from full text and answer tags separately
    - Correct if: correct SCTID found AND no wrong SCTIDs found
    - Based on snomed_sctid_reward logic from reward_funcs.py
    """
    try:
        response_text = completion
        target_sctid = str(gt_sct_id).strip()
        
        # Extract codes from answer tags if present (for answer tag metrics)
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        has_answer_tags = bool(answer_match)
        
        if has_answer_tags:
            answer_text = answer_match.group(1).strip()
            # Extract codes only from answer text
            snomed_pattern = r'\b\d{6,18}\b'
            answer_tag_codes = list(set(re.findall(snomed_pattern, answer_text)))
            
            answer_tag_correct_found = target_sctid in answer_tag_codes
            answer_tag_wrong_found = any(code != target_sctid for code in answer_tag_codes)
            answer_tag_sct_match = answer_tag_correct_found and not answer_tag_wrong_found
            answer_tag_label_match = str(gt_label).lower().strip() in answer_text.lower()
        else:
            # No answer tags found - count as incorrect (per reward function logic)
            answer_tag_codes = []
            answer_tag_correct_found = False
            answer_tag_wrong_found = False
            answer_tag_sct_match = False
            answer_tag_label_match = False
        
        # Extract codes from full text (for full text metrics)
        snomed_pattern = r'\b\d{6,18}\b'
        full_text_codes = list(set(re.findall(snomed_pattern, response_text)))
        
        # Check full text: correct if target found and no wrong codes found
        full_text_correct_found = target_sctid in full_text_codes
        full_text_wrong_found = any(code != target_sctid for code in full_text_codes)
        full_text_sct_match = full_text_correct_found and not full_text_wrong_found
        
        # Label matching (keep for compatibility, but not used in reward function)
        full_text_label_match = str(gt_label).lower().strip() in response_text.lower()
        
        # For logging: get the first code found (or target if correct)
        if full_text_codes:
            pred_sct_id = full_text_codes[0]
        else:
            pred_sct_id = "NOT_FOUND"
            
    except Exception as e:
        print(f"Error processing output: {e}")
        full_text_sct_match = False
        full_text_label_match = False
        answer_tag_sct_match = False
        answer_tag_label_match = False
        has_answer_tags = False
        pred_sct_id = "ERROR"
        response_text = f"Error: {str(e)}"
    
    result = {
        'sct_match': bool(full_text_sct_match),
        'label_match': bool(full_text_label_match),
        'gt_sct_id': str(gt_sct_id),
        'gt_label': str(gt_label),
        'pred_sct_id': pred_sct_id,
        'pred_label': str(gt_label) if full_text_label_match else "NOT_FOUND",
        'raw_output': str(response_text),
        'answer_tag_sct_match': bool(answer_tag_sct_match),
        'answer_tag_label_match': bool(answer_tag_label_match),
        'answer_tags_found': bool(has_answer_tags)
    }
    
    return result


def evaluate_snomed_accuracy(
    trainer,
    csv_path: str = "data/datasets/snomed_synthesis_dataset_test.csv",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    max_new_tokens: int = 128,
    num_return_sequences: int = 1,
    temperature: float = 0.0,  # Use deterministic generation for evaluation
    output_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate model accuracy on SNOMED dataset.
    
    Args:
        trainer: The GRPO trainer instance
        csv_path: Path to the CSV file with SNOMED data
        max_samples: Maximum number of samples to evaluate (None for all)
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum tokens to generate
        num_return_sequences: Number of sequences to generate per prompt (for pass@k)
        temperature: Temperature for generation (0.0 for deterministic)
        output_dir: Directory to save results (default: Gemma3-Finetune/results/)
    
    Returns:
        Dict with various accuracy metrics
    """
    if not os.path.exists(csv_path):
        if trainer.accelerator.is_main_process:
            print(f"Warning: Evaluation CSV not found at {csv_path}, skipping evaluation")
        return {
            "full_text_sct_id_accuracy": 0.0,
            "full_text_label_accuracy": 0.0,
            "answer_tag_sct_id_accuracy": 0.0,
            "answer_tag_label_accuracy": 0.0,
            "answer_tags_found_rate": 0.0,
            "accuracy": 0.0,
            "pass_at_k": 0.0
        }
    
    # Create dataset
    eval_dataset = SNOMEDDataset(csv_path, max_samples=max_samples)
    
    # Use DistributedSampler for proper data distribution across GPUs
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        eval_dataset,
        num_replicas=trainer.accelerator.num_processes,
        rank=trainer.accelerator.process_index,
        shuffle=False,
    )
    
    # Create dataloader with proper distribution
    # batch_size should be per-device, accelerator will handle the rest
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True,
        drop_last=False,
    )
    eval_dataloader = trainer.accelerator.prepare(eval_dataloader)
    
    # Prepare model for evaluation
    model = trainer.model
    model.eval()
    processor = trainer.processing_class
    
    # Get end_of_turn token ID for Gemma-3 models (matching grpo_trainer.py)
    stop_token_ids = [processor.eos_token_id]
    if trainer.end_of_turn_token_id is not None and trainer.end_of_turn_token_id != processor.eos_token_id:
        stop_token_ids.append(trainer.end_of_turn_token_id)
    print(f"Stop token IDs: {stop_token_ids}")
    
    # For multiple sequences, we need sampling
    effective_temperature = temperature if temperature > 0.0 else 0.7
    do_sample = num_return_sequences > 1 or temperature > 0.0
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=effective_temperature if do_sample else None,
        num_return_sequences=num_return_sequences,
        pad_token_id=processor.pad_token_id,
        eos_token_id=stop_token_ids,
    )
    
    # Metrics tracking
    total = 0
    full_text_sct_correct = 0
    full_text_label_correct = 0
    answer_tag_sct_correct = 0
    answer_tag_label_correct = 0
    answer_tags_found = 0
    correct_at_1 = 0  # First sequence correct (full text SCT match)
    correct_at_k = 0  # Any of k sequences correct (full text SCT match)
    
    all_results = []
    
    # Calculate total batches for progress bar (only on main process)
    total_batches = len(eval_dataloader) if trainer.accelerator.is_main_process else None
    dataset_name_short = os.path.basename(csv_path).replace('.csv', '')
    
    # Create progress bar only on main process
    if trainer.accelerator.is_main_process:
        pbar = tqdm(
            eval_dataloader,
            desc=f"Evaluating {dataset_name_short}",
            total=total_batches,
            unit="batch",
            disable=False
        )
    else:
        pbar = eval_dataloader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            entries = batch['entry']
            ground_truth_sct_ids = batch['sct_id']
            ground_truth_labels = batch['label']
            
            # Create prompts
            prompts = []
            for entry in entries:
                user_content = [{"type": "text", "text": str(entry)}]
                user_prompt = [{"role": "user", "content": user_content}]
                if len(SYSTEM_MESSAGE) > 0:
                    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
                    user_prompt.insert(0, system_message)
                prompts.append(user_prompt)
            
            # Process prompts
            prompts_text = []
            for p in prompts:
                text = processor.apply_chat_template(
                    p, add_generation_prompt=True, add_special_tokens=True
                )
                prompts_text.append(text.strip())
            
            # Tokenize
            prompt_inputs = processor(
                text=prompts_text,
                images=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            
            # Move to device explicitly
            device = trainer.accelerator.device
            prompt_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in prompt_inputs.items()
            }
            
            # Generate
            with trainer.compute_loss_context_manager():
                # Unwrap model for generation if needed
                from trl.models import unwrap_model_for_generation
                with unwrap_model_for_generation(
                    trainer.model_wrapped,
                    trainer.accelerator,
                    gather_deepspeed3_params=trainer.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    generated_ids = unwrapped_model.generate(
                        **prompt_inputs,
                        generation_config=generation_config
                    )
            
            # Extract prompt length and completion
            prompt_length = prompt_inputs['input_ids'].size(1)
            completion_ids = generated_ids[:, prompt_length:]
            
            # Decode completions
            completions = processor.batch_decode(completion_ids, skip_special_tokens=True)
            
            # Reshape completions: (batch_size * num_return_sequences) -> (batch_size, num_return_sequences)
            batch_size_actual = len(ground_truth_sct_ids)
            completions_reshaped = [
                completions[i * num_return_sequences : (i + 1) * num_return_sequences]
                for i in range(batch_size_actual)
            ]
            
            # Process each sample
            for sample_completions, gt_sct_id, gt_label in zip(
                completions_reshaped, ground_truth_sct_ids, ground_truth_labels
            ):
                # Process first completion for accuracy@1
                first_result = process_snomed_output(sample_completions[0], gt_sct_id, gt_label)
                all_results.append(first_result)
                
                # Update metrics from first completion
                if first_result['sct_match']:
                    full_text_sct_correct += 1
                    correct_at_1 += 1
                if first_result['label_match']:
                    full_text_label_correct += 1
                if first_result['answer_tag_sct_match']:
                    answer_tag_sct_correct += 1
                if first_result['answer_tag_label_match']:
                    answer_tag_label_correct += 1
                if first_result['answer_tags_found']:
                    answer_tags_found += 1
                
                # Check all completions for pass@k
                any_correct = False
                for completion in sample_completions:
                    result = process_snomed_output(completion, gt_sct_id, gt_label)
                    if result['sct_match']:
                        any_correct = True
                        break
                if any_correct:
                    correct_at_k += 1
                
                total += 1
            
            # Update progress bar with current metrics (only on main process)
            if trainer.accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                current_acc = correct_at_1 / total if total > 0 else 0.0
                current_pass_k = correct_at_k / total if total > 0 else 0.0
                pbar.set_postfix({
                    'acc@1': f'{current_acc:.4f}',
                    'pass@k': f'{current_pass_k:.4f}',
                    'processed': total
                })
    
    # Close progress bar if it exists
    if trainer.accelerator.is_main_process and hasattr(pbar, 'close'):
        pbar.close()
    
    # Gather results from all processes using accelerator
    metrics_tensors = {
        'full_text_sct': torch.tensor([full_text_sct_correct], device=trainer.accelerator.device, dtype=torch.long),
        'full_text_label': torch.tensor([full_text_label_correct], device=trainer.accelerator.device, dtype=torch.long),
        'answer_tag_sct': torch.tensor([answer_tag_sct_correct], device=trainer.accelerator.device, dtype=torch.long),
        'answer_tag_label': torch.tensor([answer_tag_label_correct], device=trainer.accelerator.device, dtype=torch.long),
        'answer_tags_found': torch.tensor([answer_tags_found], device=trainer.accelerator.device, dtype=torch.long),
        'correct_at_1': torch.tensor([correct_at_1], device=trainer.accelerator.device, dtype=torch.long),
        'correct_at_k': torch.tensor([correct_at_k], device=trainer.accelerator.device, dtype=torch.long),
        'total': torch.tensor([total], device=trainer.accelerator.device, dtype=torch.long),
    }
    
    # Gather across all processes
    gathered_metrics = {
        k: trainer.accelerator.gather_for_metrics(v) for k, v in metrics_tensors.items()
    }
    
    # Sum across all processes
    total_full_text_sct = gathered_metrics['full_text_sct'].sum().item()
    total_full_text_label = gathered_metrics['full_text_label'].sum().item()
    total_answer_tag_sct = gathered_metrics['answer_tag_sct'].sum().item()
    total_answer_tag_label = gathered_metrics['answer_tag_label'].sum().item()
    total_answer_tags_found = gathered_metrics['answer_tags_found'].sum().item()
    total_correct_at_1 = gathered_metrics['correct_at_1'].sum().item()
    total_correct_at_k = gathered_metrics['correct_at_k'].sum().item()
    total_samples = gathered_metrics['total'].sum().item()
    
    # Calculate accuracies
    full_text_sct_accuracy = total_full_text_sct / total_samples if total_samples > 0 else 0.0
    full_text_label_accuracy = total_full_text_label / total_samples if total_samples > 0 else 0.0
    answer_tag_sct_accuracy = total_answer_tag_sct / total_samples if total_samples > 0 else 0.0
    answer_tag_label_accuracy = total_answer_tag_label / total_samples if total_samples > 0 else 0.0
    answer_tags_found_rate = total_answer_tags_found / total_samples if total_samples > 0 else 0.0
    accuracy_at_1 = total_correct_at_1 / total_samples if total_samples > 0 else 0.0
    accuracy_at_k = total_correct_at_k / total_samples if total_samples > 0 else 0.0
    
    results_dict = {
        "full_text_sct_id_accuracy": full_text_sct_accuracy,
        "full_text_label_accuracy": full_text_label_accuracy,
        "answer_tag_sct_id_accuracy": answer_tag_sct_accuracy,
        "answer_tag_label_accuracy": answer_tag_label_accuracy,
        "answer_tags_found_rate": answer_tags_found_rate,
        "accuracy": accuracy_at_1,
        "pass_at_k": accuracy_at_k,
    }
    
    if trainer.accelerator.is_main_process:
        print(f"\nSNOMED Evaluation ({csv_path}):")
        print(f"  Full Text SCT ID Accuracy:  {full_text_sct_accuracy:.4f} ({total_full_text_sct}/{total_samples})")
        print(f"  Full Text Label Accuracy:   {full_text_label_accuracy:.4f} ({total_full_text_label}/{total_samples})")
        print(f"  Answer Tag SCT ID Accuracy:  {answer_tag_sct_accuracy:.4f} ({total_answer_tag_sct}/{total_samples})")
        print(f"  Answer Tag Label Accuracy:   {answer_tag_label_accuracy:.4f} ({total_answer_tag_label}/{total_samples})")
        print(f"  Answer Tags Found Rate:      {answer_tags_found_rate:.4f} ({total_answer_tags_found}/{total_samples})")
        print(f"  Accuracy@1:                  {accuracy_at_1:.4f} ({total_correct_at_1}/{total_samples})")
        print(f"  Pass@{num_return_sequences}: {accuracy_at_k:.4f} ({total_correct_at_k}/{total_samples})")
        
        # Save results to file
        if output_dir is None:
            # Default to Gemma3-Finetune/results/
            # __file__ is src/train/evaluate_snomed.py, so go up 3 levels to get to Gemma3-Finetune/
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(base_dir, "results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename from CSV path
        csv_basename = os.path.basename(csv_path).replace('.csv', '')
        checkpoint_name = getattr(trainer.args, 'output_dir', 'checkpoint')
        if checkpoint_name:
            checkpoint_name = os.path.basename(checkpoint_name.rstrip('/'))
        else:
            checkpoint_name = 'unknown'
        
        output_file = os.path.join(output_dir, f"snomed_evaluation_{csv_basename}_{checkpoint_name}.json")
        
        # Save detailed results
        save_dict = {
            'metrics': results_dict,
            'csv_path': csv_path,
            'checkpoint': checkpoint_name,
            'num_samples': total_samples,
            'num_return_sequences': num_return_sequences,
            'details': all_results[:1000] if len(all_results) > 1000 else all_results  # Limit details to avoid huge files
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        
        print(f"  Results saved to {output_file}")
    
    return results_dict
