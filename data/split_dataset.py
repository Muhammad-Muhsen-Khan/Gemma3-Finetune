#!/usr/bin/env python
"""
Split SNOMED findings dataset into train and test sets with stratified sampling by label.
Creates a 95/5 train/test split.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(
    input_csv_path: str | Path,
    train_size: float = 0.95,
    random_state: int = 42,
    stratify_by: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets with stratified sampling.
    
    Args:
        input_csv_path: Path to input CSV file
        train_size: Proportion of data for training (default: 0.95)
        random_state: Random seed for reproducibility (default: 42)
        stratify_by: Column name to stratify by (default: "label")
    
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"Loading dataset from: {input_csv_path}")
    df = pd.read_csv(input_csv_path, encoding='utf-8')
    
    print(f"Total samples: {len(df)}")
    print(f"Unique labels: {df[stratify_by].nunique()}")
    
    # Check if stratify column exists
    if stratify_by not in df.columns:
        raise ValueError(f"Column '{stratify_by}' not found in dataset")
    
    # Remove rows with missing labels (needed for stratification)
    df_clean = df.dropna(subset=[stratify_by])
    if len(df_clean) < len(df):
        print(f"Removed {len(df) - len(df_clean)} rows with missing {stratify_by} values")
    
    # Perform stratified split
    print(f"\nSplitting dataset ({train_size*100:.0f}% train, {(1-train_size)*100:.0f}% test)...")
    train_df, test_df = train_test_split(
        df_clean,
        train_size=train_size,
        test_size=1 - train_size,
        random_state=random_state,
        stratify=df_clean[stratify_by],
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Train labels: {train_df[stratify_by].nunique()} unique")
    print(f"Test labels: {test_df[stratify_by].nunique()} unique")
    
    return train_df, test_df


if __name__ == "__main__":
    # Set paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / "datasets" / "snomed_synthesis_dataset_set_C.csv"
    
    # Check if input file exists
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    # Split dataset
    train_df, test_df = split_dataset(
        input_csv,
        train_size=0.95,
        random_state=42,
        stratify_by="label",
    )
    
    # Generate output filenames
    input_stem = input_csv.stem
    train_output = script_dir / "datasets" / f"{input_stem}_train.csv"
    test_output = script_dir / "datasets" / f"{input_stem}_test.csv"
    
    # Save train and test sets
    print(f"\nSaving train set to: {train_output}")
    train_df.to_csv(train_output, index=False, encoding='utf-8')
    
    print(f"Saving test set to: {test_output}")
    test_df.to_csv(test_output, index=False, encoding='utf-8')
    
    print("\nDataset split completed successfully!")
    print(f"Train: {train_output}")
    print(f"Test: {test_output}")
