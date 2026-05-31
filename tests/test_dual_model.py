import torch
from ifg26.models.dual_model import DualModalityMLP

def test_dual_modality_forward():
    """Verify forward pass with split and concatenated inputs."""
    device = "cpu"
    model = DualModalityMLP(
        ligand_dim=2048,
        protein_dim=1280,
        n_proteins=2,
        ligand_hidden=[64],
        protein_hidden=[64],
        fusion_hidden=[32]
    ).to(device)
    
    batch_size = 4
    x_lig = torch.randn(batch_size, 2048).to(device)
    x_prot = torch.randn(batch_size, 2560).to(device) # 1280 * 2
    
    model.eval()
    
    # Test 1: Split input
    out1 = model(x_lig, x_prot)
    assert out1.shape == (batch_size,)
    assert (out1 >= 0).all() and (out1 <= 1).all()
    
    # Test 2: Concatenated input
    x_concat = torch.cat([x_lig, x_prot], dim=1)
    out2 = model(x_concat)
    assert out2.shape == (batch_size,)
    assert torch.allclose(out1, out2)
    
    print("DualModalityMLP forward test: PASS")

if __name__ == "__main__":
    test_dual_modality_forward()
