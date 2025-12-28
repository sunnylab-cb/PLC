import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weight for the positive class (0 < alpha < 1).
            gamma (float): Focusing parameter.
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (probabilities) from the model (after Sigmoid).
            targets: Ground truth labels (0 or 1).
        """
        # Ensure inputs are probabilities (already applied Sigmoid in model)
        # However, for numerical stability, usually it's better to work with logits.
        # But our model outputs sigmoid. Let's work with that for now, assuming inputs are in [0, 1].
        
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Alpha weighting: alpha for class 1, (1-alpha) for class 0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
