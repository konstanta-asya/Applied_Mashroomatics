import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces the relative loss for well-classified examples,
    putting more focus on hard, misclassified examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SeeSawLoss(nn.Module):
    """
    Seesaw Loss for long-tailed recognition.
    Dynamically re-balances gradients based on class distribution.
    """
    def __init__(self, class_counts, p=0.8, q=2.0, eps=1e-2):
        super().__init__()
        self.p = p
        self.q = q
        self.eps = eps

        # Compute class weights from counts
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        self.register_buffer('class_counts', class_counts)

    def forward(self, inputs, targets):
        # Standard cross entropy as base
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute mitigation factor based on class frequencies
        target_counts = self.class_counts[targets]
        all_counts = self.class_counts.sum()

        # Seesaw reweighting
        weight = (all_counts / (target_counts + self.eps)) ** self.p
        weight = torch.clamp(weight, max=10.0)  # Prevent extreme weights

        weighted_loss = weight * ce_loss
        return weighted_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing for better generalization.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()