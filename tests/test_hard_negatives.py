import pandas as pd
from ifg26.splits.hard_negatives import build_hard_near_tier

def test_hard_near_generation():
    """Verify that hard negatives are similar to parents and match properties."""
    # Dummy data
    pos_df = pd.DataFrame({
        'ligand_smiles': ['Cc1ccccc1', 'C1CCCCC1'],
        'e3_uniprot_id': ['P1', 'P2'],
        'target_uniprot_id': ['T1', 'T2']
    })
    
    universe_df = pd.DataFrame({
        'smiles': ['Cc1cccc(C)c1', 'C1CCCC(C)C1', 'c1ccccc1', 'CCCC'],
        'mw': [106.1, 98.1, 78.1, 58.1],
        'logp': [2.2, 2.5, 2.1, 2.0]
    })
    
    # Run
    hard_df = build_hard_near_tier(pos_df, universe_df, n_per_pos=2)
    
    print(f"Generated {len(hard_df)} hard-near samples.")
    print(hard_df[['ligand_smiles', 'parent_positive']])
    
    assert len(hard_df) > 0
    assert 'hard_near' in hard_df['source_type'].values

if __name__ == "__main__":
    test_hard_near_generation()
