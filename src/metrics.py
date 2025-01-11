import numpy as np
import torch
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



def aggregate_prediction_labels(label_predictions, true_labels, chunk_length, overlap):
    """
    Aggregate predictions and true labels for overlapping chunks.

    Args:
        label_predictions (list of np.array): List of arrays containing predicted labels for each chunk.
        true_labels (list of np.array): List of arrays containing true labels for each chunk.
        chunk_length (int): Length of each chunk.
        overlap (int): Overlap between consecutive chunks.

    Returns:
        aggregated_predictions (np.array): Aggregated predictions for the entire sequence.
        aggregated_labels (np.array): Aggregated true labels for the entire sequence.
    """
    assert len(label_predictions) == len(true_labels), "Predictions and labels must have the same number of chunks."

    # Step size and total sequence length
    step = chunk_length - overlap
    sequence_length = (len(label_predictions) - 1) * step + chunk_length

    # Initialize arrays for aggregated predictions and labels
    aggregated_predictions = [[] for _ in range(sequence_length)]
    aggregated_labels = np.zeros((sequence_length))
    # Populate aggregated labels
    for i in range(sequence_length):
        chunk_idx = i // step
        offset = i % step
        if chunk_idx >= len(true_labels):
            chunk_idx = len(true_labels) - 1
            offset = i % chunk_length
        aggregated_labels[i] = true_labels[chunk_idx][offset]

    # Populate aggregated predictions
    for i, chunk in enumerate(label_predictions):
        for j, pred in enumerate(chunk):
            index = j + i * step
            if index < sequence_length:
                aggregated_predictions[index].append(pred.item() if isinstance(pred, np.ndarray) else pred)


    # Majority voting for predictions
    aggregated_predictions = np.array([
        Counter(map(int, pred_list)).most_common(1)[0][0] if pred_list else -1
        for pred_list in aggregated_predictions
    ])

    return aggregated_predictions, aggregated_labels


def compute_metrics(y_true, y_pred):
    """
    Compute metrics for classification.

    Args:
        y_true: Ground truth labels (class indices as a numpy array or tensor).
        y_pred: Predicted labels (class indices as a list, numpy array, or tensor).

    Returns:
        precision, recall, f1, accuracy: Computed metrics.
    """
    if(len(y_pred)==0):
      return (0,0,0,0)
    # Convert y_pred to a NumPy array if it's a list
    if isinstance(y_pred, list):
        y_pred = np.concatenate(y_pred, axis=0)

    # Convert y_true to a NumPy array if it's not already
    if isinstance(y_true, list):
        y_true = np.concatenate(y_true, axis=0)

    # Ensure inputs are NumPy arrays for compatibility with scikit-learn metrics
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    # Compute metrics
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy
