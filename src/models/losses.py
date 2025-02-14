import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

def regular_loss(y_pred, y_true, alpha=(1.0, 1.0, 1.0, 1.0, 1.0)):
    """
    y_pred : tuple of predictions (logits)
    y_true : tuple of ground truth labels
    """
    y_pred_stroke, y_pred_player, y_pred_type, y_pred_role, y_pred_impact = y_pred
    y_true_stroke = y_true[:, :, 0]
    y_true_player = y_true[:, :, 1]
    y_true_type = y_true[:, :, 2]
    y_true_role = y_true[:, :, 3]
    y_true_impact = y_true[:, :, 4]
    mask_stroke = (y_true_stroke != -1).squeeze(-1)

    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    loss_stroke = bce_with_logits_loss(y_pred_stroke[mask_stroke], y_true_stroke[mask_stroke]) if mask_stroke.sum() > 0 else torch.tensor(0.0, device=y_pred_stroke.device)

    mask = (y_true_stroke == 1).squeeze(-1)

    # Player and Type tasks (No weighting applied here)
    loss_player = F.cross_entropy(y_pred_player[mask], y_true_player[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_player.device)
    loss_type = F.cross_entropy(y_pred_type[mask], y_true_type[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_type.device)
    loss_role = F.cross_entropy(y_pred_role[mask], y_true_role[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_role.device)
    loss_impact = F.cross_entropy(y_pred_impact[mask], y_true_impact[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_role.device)

    total_loss = (
        alpha[0] * loss_stroke +
        alpha[1] * loss_player +
        alpha[2] * loss_type +
        alpha[3] * loss_role +
        alpha[4] * loss_impact
    )
    return total_loss


def weighted_loss(y_pred, y_true, alpha=(1.0, 1.0, 1.0, 1.0, 1.0)):
    """
    y_pred : tuple of predictions (logits)
    y_true : tuple of ground truth labels
    """
    y_pred_stroke, y_pred_player, y_pred_type, y_pred_role, y_pred_impact = y_pred
    y_true_stroke = y_true[:, :, 0]
    y_true_player = y_true[:, :, 1]
    y_true_type = y_true[:, :, 2]
    y_true_role = y_true[:, :, 3]
    y_true_impact = y_true[:, :, 4]

    mask_stroke = (y_true_stroke != -1).squeeze(-1)
    #Weights are calculated based on all data using the following formula w_i = nb_total_samples / (nb_classes * nb_samples_i)
    weight = torch.zeros_like(y_true_stroke)
    weight[y_true_stroke == 1] = 48.0 
    weight[y_true_stroke == 0] = 0.5
    weight = weight[mask_stroke].squeeze(-1)

  

    bce_with_logits_loss = nn.BCEWithLogitsLoss(weight=weight)
    loss_stroke = bce_with_logits_loss(y_pred_stroke[mask_stroke].squeeze(-1), y_true_stroke[mask_stroke].squeeze(-1)) if mask_stroke.sum() > 0 else torch.tensor(0.0, device=y_pred_stroke.device)

    mask = (y_true_stroke == 1).squeeze(-1)

    # Player and Type tasks (No weighting applied here almost balanced)
    loss_player = F.cross_entropy(y_pred_player[mask], y_true_player[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_player.device)
    loss_type = F.cross_entropy(y_pred_type[mask], y_true_type[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_type.device)

    weights_role = torch.tensor([1.63, 3.41, 0.47], dtype=torch.float32, device=y_pred_role.device)

    loss_role = F.cross_entropy(y_pred_role[mask], y_true_role[mask], weight=weights_role) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_role.device)
    
    weights_impact = torch.tensor([8.71, 1.42, 34.7, 0.31], dtype=torch.float32, device=y_pred_role.device)
  
    loss_impact = F.cross_entropy(y_pred_impact[mask], y_true_impact[mask], weight=weights_impact) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_role.device)

    total_loss = (
        alpha[0] * loss_stroke +
        alpha[1] * loss_player +
        alpha[2] * loss_type +
        alpha[3] * loss_role +
        alpha[4] * loss_impact
    )
    return total_loss


def multivariate_weighted_loss(y_pred, y_true, log_vars):
    """
    y_pred : tuple of predictions (logits)
    y_true : tuple of ground truth labels
    """
    y_pred_stroke, y_pred_player, y_pred_type, y_pred_role, y_pred_impact = y_pred
    y_true_stroke = y_true[:, :, 0]
    y_true_player = y_true[:, :, 1]
    y_true_type = y_true[:, :, 2]
    y_true_role = y_true[:, :, 3]
    y_true_impact = y_true[:, :, 4]

    mask_stroke = (y_true_stroke != -1).squeeze(-1)
    #Weights are calculated based on all data using the following formula w_i = nb_total_samples / (nb_classes * nb_samples_i)
    weight = torch.zeros_like(y_true_stroke)
    weight[y_true_stroke == 1] = 48.0 
    weight[y_true_stroke == 0] = 0.5
    weight = weight[mask_stroke].squeeze(-1)

  

    bce_with_logits_loss = nn.BCEWithLogitsLoss(weight=weight)
    loss_stroke = bce_with_logits_loss(y_pred_stroke[mask_stroke].squeeze(-1), y_true_stroke[mask_stroke].squeeze(-1)) if mask_stroke.sum() > 0 else torch.tensor(0.0, device=y_pred_stroke.device)

    mask = (y_true_stroke == 1).squeeze(-1)

    # Player and Type tasks (No weighting applied here almost balanced)
    loss_player = F.cross_entropy(y_pred_player[mask], y_true_player[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_player.device)
    loss_type = F.cross_entropy(y_pred_type[mask], y_true_type[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_type.device)

    weights_role = torch.tensor([1.63, 3.41, 0.47], dtype=torch.float32, device=y_pred_role.device)

    loss_role = F.cross_entropy(y_pred_role[mask], y_true_role[mask], weight=weights_role) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_role.device)
    
    weights_impact = torch.tensor([8.71, 1.42, 34.7, 0.31], dtype=torch.float32, device=y_pred_role.device)
  
    loss_impact = F.cross_entropy(y_pred_impact[mask], y_true_impact[mask], weight=weights_impact) if mask.sum() > 0 else torch.tensor(0.0, device=y_pred_role.device)

    total_loss = (
        (1 / (2 * torch.exp(log_vars[0]))) * loss_stroke + log_vars[0] +
        (1 / (2 * torch.exp(log_vars[1]))) * loss_player + log_vars[1] +
        (1 / (2 * torch.exp(log_vars[2]))) * loss_type + log_vars[2] +
        (1 / (2 * torch.exp(log_vars[3]))) * loss_role + log_vars[3] +
        (1 / (2 * torch.exp(log_vars[4]))) * loss_impact + log_vars[4]
    )
    return total_loss

