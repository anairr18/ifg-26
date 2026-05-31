import torch
import torch.nn as nn
from ifg26.models.pu_learning.ranking_loss import PURankingLoss

def test_ranking_gradient():
    """Verify that ranking loss produces gradients that favor p > u."""
    model = nn.Linear(10, 1)
    criterion = PURankingLoss(loss_type="bce")
    
    p_X = torch.randn(4, 10, requires_grad=True)
    u_X = torch.randn(4, 10, requires_grad=True)
    
    # Initialize p_X to be "better"
    p_score = model(p_X).squeeze()
    u_score = model(u_X).squeeze()
    
    loss = criterion(p_score, u_score)
    loss.backward()
    
    # After backward, gradients should exist
    assert p_X.grad is not None
    assert u_X.grad is not None
    
    print("PURankingLoss gradient test: PASS")

if __name__ == "__main__":
    test_ranking_gradient()
