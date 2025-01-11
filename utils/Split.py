
"""
split.py

Author: JORMANA
Date: 2024-11-25

"""

def calculate_file_lengths(feature_files):
    """
    Calculate the number of rows in each feature file.
    Args:
        feature_files: List of file paths to feature CSVs.
    Returns:
        List of lengths corresponding to each file.
    """
    file_lengths = []
    for file in feature_files:
        with open(file) as f:
            file_lengths.append(sum(1 for _ in f) - 1)  
    return file_lengths

def proportional_split(feature_files, target_files, file_lengths, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split files proportionally by their lengths into train, validation, and test sets.
    Args:
        feature_files: List of feature file paths.
        target_files: List of target file paths.
        file_lengths: List of file lengths (row counts).
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
    Returns:
        Tuple: (train_features, train_targets, val_features, val_targets, test_features, test_targets)
    """
    assert len(feature_files) == len(target_files) == len(file_lengths), "Mismatch in file lists and lengths."
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    total_rows = sum(file_lengths)
    train_threshold = total_rows * train_ratio
    val_threshold = total_rows * (train_ratio + val_ratio)

    train_features, val_features, test_features = [], [], []
    train_targets, val_targets, test_targets = [], [], []
    cumulative_rows = 0

    for i, length in enumerate(file_lengths):
        if cumulative_rows < train_threshold and cumulative_rows + length <= train_threshold:
            train_features.append(feature_files[i])
            train_targets.append(target_files[i])
        elif cumulative_rows < val_threshold and cumulative_rows + length <= val_threshold:
            val_features.append(feature_files[i])
            val_targets.append(target_files[i])
        else:
            test_features.append(feature_files[i])
            test_targets.append(target_files[i])
        cumulative_rows += length

    # Check if the splits are empty or imbalanced
    assert train_features, "Training set is empty!"
    assert val_features, "Validation set is empty!"
    assert test_features, "Test set is empty!"

    return train_features, train_targets, val_features, val_targets, test_features, test_targets
