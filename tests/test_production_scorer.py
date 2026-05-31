import torch
import numpy as np
from ifg26.inference.scorer import IFG26Scorer
from ifg26.models.dual_model import DualModalityMLP

def test_scorer_forward():
    """Verify that IFG26Scorer correctly performs batch inference."""
    # 1. Setup local "model" for testing (skipping weight loading for speed)
    model = DualModalityMLP(ligand_dim=2048, protein_dim=320, n_proteins=2) # 320 for t6_8M
    torch.save(model.state_dict(), "tmp_model.pt")
    
    # 2. Initialize Scorer (using 8M model for speed)
    scorer = IFG26Scorer("tmp_model.pt", device="cpu", esm_model_name="esm2_t6_8M_UR50D")
    
    # 3. Dummy SMILES and Protein Seq
    smiles_list = ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    e3_seq = "MKTVRQERLKSIVRILERS"
    target_seq = "MKTVRQERLKSIVRILERS"
    
    # 4. Score
    scores = scorer.score(smiles_list, e3_seq, target_seq)
    
    print(f"Scores: {scores}")
    assert len(scores) == 2
    assert (scores >= 0).all() and (scores <= 1).all()
    
    print("IFG26Scorer inference test: PASS")

if __name__ == "__main__":
    test_scorer_forward()
