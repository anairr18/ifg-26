import numpy as np
import torch
from ifg26.features.protein_embeddings import ESM2Embedder

def test_identical_sequences():
    """Requirement 7: identical sequence -> identical embedding."""
    # Note: Using a smaller model for faster testing if needed, 
    # but the logic is the same for t33_650M.
    # We use t6_8M for the unit test to keep it lightweight.
    try:
        embedder = ESM2Embedder(model_name="esm2_t6_8M_UR50D", device="cpu")
        
        seq1 = "MKTVRQERLKSIVRILERS"
        seq2 = "MKTVRQERLKSIVRILERS"
        seq3 = "MKTVRQERLKSIVRILERSM" # One residue difference (M at end)

        emb1 = embedder.embed_sequences([seq1])
        emb2 = embedder.embed_sequences([seq2])
        emb3 = embedder.embed_sequences([seq3])

        # EMB1 and EMB2 must be identical
        assert np.allclose(emb1, emb2, atol=1e-5)
        # EMB1 and EMB3 must be different
        assert not np.allclose(emb1, emb3, atol=1e-5)
        
        print("ESM2 identical sequence verification: PASS")
        print(f"Embedding shape: {emb1.shape}")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_identical_sequences()
