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


def compute_metrics(y_true, y_pred, task=None, tolerance=2):
    """
    Compute metrics for classification with an optional tolerance for stroke detection.

    Args:
        y_true: Ground truth labels (NumPy array or tensor).
        y_pred: Predicted labels (list, NumPy array, or tensor).
        task: If "stroke", applies tolerance to predictions.
        tolerance: Margin of tolerance for stroke predictions.

    Returns:
        precision, recall, f1, accuracy: Computed metrics.
    """
    if len(y_pred) == 0:
        return 0, 0, 0, 0

    # Convert to NumPy arrays if necessary
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    if task == "stroke":
        y_pred_adjusted = np.zeros_like(y_pred)
        for i in range(len(y_true)):
            if y_true[i] == 1:
                # Define the tolerance window
                start = max(0, i - tolerance)
                end = min(len(y_pred), i + tolerance + 1)
                # Check if there is exactly one predicted event in this window
                if np.sum(y_pred[start:end]) == 1:
                    y_pred_adjusted[i] = 1
        y_pred = y_pred_adjusted

    # Compute metrics
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy

def select_highest_score_stroke(batch_scores, score_threshold=0.5):
    """
    Refine stroke predictions for each element in the batch by selecting 
    only the stroke with the highest score in each group of consecutive frames.

    Args:
        batch_scores (np.ndarray): Predicted stroke scores (batch_size x sequence_length).
        score_threshold (float): Minimum score to consider a frame as a stroke candidate.

    Returns:
        np.ndarray: Array with the refined stroke predictions (binary values),
                    same shape as `batch_scores`.
    """
    refined_batch_predictions = np.zeros_like(batch_scores, dtype=int)
    
    for batch_idx in range(batch_scores.shape[0]):  
        scores = batch_scores[batch_idx]
        above_threshold = scores > score_threshold
        refined_predictions = np.zeros_like(scores, dtype=int)
        indices = np.where(above_threshold)[0]
        print(indices)
        if len(indices) == 0:  
            refined_batch_predictions[batch_idx] = refined_predictions
            continue

        current_group = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:  
                current_group.append(indices[i])
            else:
                best_frame = current_group[np.argmax(scores[current_group])]
                refined_predictions[best_frame] = 1
                current_group = [indices[i]]
        
        if current_group:
            best_frame = current_group[np.argmax(scores[current_group])]
            refined_predictions[best_frame] = 1
        
        refined_batch_predictions[batch_idx] = refined_predictions

    return refined_batch_predictions




