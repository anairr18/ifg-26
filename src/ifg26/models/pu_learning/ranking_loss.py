"""
ranking_loss.py
===============
Pairwise ranking losses for Positive-Unlabeled (PU) learning.
Implements MarginRankingLoss and RankNet objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PURankingLoss(nn.Module):
    """
    Pairwise ranking loss for PU learning.
    Expects (positive_scores, unlabeled_scores) pairs or batches.
    """
    def __init__(self, margin: float = 1.0, loss_type: str = "margin"):
        super().__init__()
        self.margin = margin
        self.loss_type = loss_type
        
    def forward(self, pos_scores: torch.Tensor, unl_scores: torch.Tensor, pi: float = 0.05) -> torch.Tensor:
        """
        Computes the pairwise ranking loss between positives and unlabeled.
        Args:
            pos_scores: (batch_size,) scores for known positives
            unl_scores: (batch_size,) scores for unlabeled pool
            pi: Class prior (used to weight unlabeled contributions if needed)
        """
        if self.loss_type == "margin":
            # Target = 1 (pos > unl)
            target = torch.ones_like(pos_scores)
            loss = F.margin_ranking_loss(pos_scores, unl_scores, target, margin=self.margin)
        elif self.loss_type == "bce":
            # RankNet Style: sigmoid(pos - unl) should be 1
            diff = pos_scores - unl_scores
            loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss

def compute_ndcg(y_true, y_score, k=None):
    """Compute Normalized Discounted Cumulative Gain."""
    from sklearn.metrics import ndcg_score
    # Reshape for sklearn [n_samples, n_classes]
    y_true = np.asarray(y_true).reshape(1, -1)
    y_score = np.asarray(y_score).reshape(1, -1)
    return ndcg_score(y_true, y_score, k=k)
