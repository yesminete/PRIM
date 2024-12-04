import torch
import torch.nn.functional as F
import torch.nn as nn

def loss(y_pred, y_true, alpha=(1.0, 1.0, 1.0,1.0,1.0)):
  """
  y_pred : tuple of predictions (logits)
  y_true : tuple of ground truth labels
  """
  y_pred_stroke, y_pred_player, y_pred_type, y_pred_role, y_pred_impact = y_pred
  y_true_stroke = y_true[:, :, 0].unsqueeze(-1)
  y_true_player = y_true[:, :, 1:3]
  y_true_type = y_true[:, :, 3:5]
  y_true_role = torch.stack([y_true[:, :, 5], y_true[:, :, 6], y_true[:, :, 5] + y_true[:, :, 6]], dim=-1)
  y_true_impact = torch.cat([y_true[:, :, 7:], torch.sum(y_true[:, :, 7:], dim=-1, keepdim=True)], dim=-1)


  bce_with_logits_loss = nn.BCEWithLogitsLoss()
  loss_stroke = bce_with_logits_loss(y_pred_stroke, y_true_stroke)


  mask = (y_true_stroke == 1).squeeze(-1)
  loss_player = F.cross_entropy(
        y_pred_player[mask], y_true_player[mask]
  ) if mask.sum() > 0 else 0.0

  loss_type = F.cross_entropy(
        y_pred_type[mask], y_true_type[mask]
  ) if mask.sum() > 0 else 0.0

  loss_role = F.cross_entropy(
        y_pred_role[mask], y_true_role[mask]
  ) if mask.sum() > 0 else 0.0


  loss_impact = F.cross_entropy(
        y_pred_impact[mask], y_true_impact[mask]
  ) if mask.sum() > 0 else 0.0

  total_loss = alpha[0] * loss_stroke + alpha[1] * loss_player + alpha[2] * loss_type + alpha[3] * loss_role + alpha[4] * loss_impact

  return total_loss
