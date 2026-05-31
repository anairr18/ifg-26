import pandas as pd
import numpy as np
import os
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm.auto import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from molecule_tracker import MoleculeTracker

# Ladder config
STRICT = {'MW': 0.05, 'cLogP': 0.4, 'TPSA': 8, 'HBD': 1, 'HBA': 1, 'RotB': 1, 'SimMin': 0.40, 'SimMax': 0.70}
R1 = {'MW': 0.05, 'cLogP': 0.4, 'TPSA': 8, 'HBD': 1, 'HBA': 1, 'RotB': 1, 'SimMin': 0.30, 'SimMax': 0.75}
R2 = {'MW': 0.07, 'cLogP': 0.6, 'TPSA': 12, 'HBD': 1, 'HBA': 1, 'RotB': 1, 'SimMin': 0.30, 'SimMax': 0.75}
R3 = {'MW': 0.07, 'cLogP': 0.6, 'TPSA': 12, 'HBD': 2, 'HBA': 2, 'RotB': 2, 'SimMin': 0.30, 'SimMax': 0.75}
R4 = {'MW': 0.07, 'cLogP': 0.6, 'TPSA': 12, 'HBD': 2, 'HBA': 2, 'RotB': 2, 'SimMin': 0.20, 'SimMax': 0.80}

def parse_desc(desc_series):
    return pd.DataFrame([json.loads(x) if isinstance(x, str) else x for x in desc_series])

def get_physchem_mask(p_row, pool_desc, P_DICT):
    mw_bound = p_row['MW'] * P_DICT['MW']
    mask = (pool_desc['FormalCharge'] == p_row['FormalCharge']) & \
           (pool_desc['MW'] >= p_row['MW'] - mw_bound) & \
           (pool_desc['MW'] <= p_row['MW'] + mw_bound) & \
           (pool_desc['cLogP'] >= p_row['cLogP'] - P_DICT['cLogP']) & \
           (pool_desc['cLogP'] <= p_row['cLogP'] + P_DICT['cLogP']) & \
           (pool_desc['TPSA'] >= p_row['TPSA'] - P_DICT['TPSA']) & \
           (pool_desc['TPSA'] <= p_row['TPSA'] + P_DICT['TPSA']) & \
           (pool_desc['HBD'] >= p_row['HBD'] - P_DICT['HBD']) & \
           (pool_desc['HBD'] <= p_row['HBD'] + P_DICT['HBD']) & \
           (pool_desc['HBA'] >= p_row['HBA'] - P_DICT['HBA']) & \
           (pool_desc['HBA'] <= p_row['HBA'] + P_DICT['HBA']) & \
           (pool_desc['RotB'] >= p_row['RotB'] - P_DICT['RotB']) & \
           (pool_desc['RotB'] <= p_row['RotB'] + P_DICT['RotB'])
    return mask

from rdkit.Chem import rdFingerprintGenerator

def compute_fps(smiles_list, tracker, source_file, stage):
    fps = []
    valid_idx = []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for i, sm in enumerate(smiles_list):
        mol = tracker.parse(sm, source_file=source_file, stage=stage, record_id=str(i))
        if mol:
            fp = mfpgen.GetFingerprint(mol)
            fps.append(fp)
            valid_idx.append(i)
    return fps, valid_idx

def main():
    print("Loading data...")
    decoy_df = pd.read_parquet('data/negatives/decoy_pool.parquet')
    pool_p_df = pd.read_parquet('data/pu/pool_P_scaffold.parquet')
    ligands_df = pd.read_parquet('data/features/ligands_physchem.parquet')

    # Load smiles map from raw
    print("Loading raw SMILES strings to match InChIKeys...")
    smiles_map = {}
    
    # 1. MGTbind
    if os.path.exists('data/raw/mgtbind/mgtbind_compounds.csv'):
        mgt = pd.read_csv('data/raw/mgtbind/mgtbind_compounds.csv')
        if 'smiles' in mgt.columns:
            for sm in mgt['smiles'].dropna().unique():
                try:
                    mol = Chem.MolFromSmiles(sm)
                    if mol: smiles_map[Chem.MolToInchiKey(mol)] = sm
                except: pass
                
    # 2. MGDB
    if os.path.exists('data/raw/mgdb/mgdb_compounds.csv'):
        mgdb = pd.read_csv('data/raw/mgdb/mgdb_compounds.csv', skiprows=1)
        if 'Smiles' in mgdb.columns:
            for sm in mgdb['Smiles'].dropna().unique():
                try:
                    mol = Chem.MolFromSmiles(sm)
                    if mol: smiles_map[Chem.MolToInchiKey(mol)] = sm
                except: pass

    # 3. integrity report
    if os.path.exists('dataset/smiles_integrity_report.csv'):
        ir = pd.read_csv('dataset/smiles_integrity_report.csv')
        if 'smiles' in ir.columns:
            for sm in ir['smiles'].dropna().unique():
                try:
                    mol = Chem.MolFromSmiles(sm)
                    if mol: smiles_map[Chem.MolToInchiKey(mol)] = sm
                except: pass

    # Convert P to use correct smiles and physchem
    p_merged = pool_p_df.merge(ligands_df, left_on='ligand_inchikey', right_on='inchi_key', how='inner')
    
    # Map smiles
    p_merged['smiles'] = p_merged['ligand_inchikey'].map(smiles_map)
    # drop rows where we couldn't find a smile
    p_merged = p_merged.dropna(subset=['smiles']).reset_index(drop=True)
    print(f"Recovered SMILES for {len(p_merged)} out of {len(pool_p_df)} P ligands.")
    
    # Pre-calculate structures for Decoy pool
    # The decoy pool comes with 'rdkit_desc' and 'murcko' explicitly!
    print("Preparing Decoy descriptions...")
    decoy_desc = parse_desc(decoy_df['rdkit_desc'])
    tracker = MoleculeTracker(name="phase4E_generate_strict_pmd")
    decoy_fps, decoy_valid_idx = compute_fps(decoy_df['smiles'].tolist(), tracker, 'decoy')
    decoy_df = decoy_df.iloc[decoy_valid_idx].reset_index(drop=True)
    decoy_desc = decoy_desc.iloc[decoy_valid_idx].reset_index(drop=True)
    decoy_scaffolds = decoy_df['murcko'].values
    
    # We need to construct P physchem dict as well
    # Re-calculate or map existing? ligands_physchem has raw properties
    # P might missing 'murcko' if not in ligands, let's compute it quickly for P
    p_smiles = p_merged['smiles'].tolist()
    p_fps, p_valid_idx = compute_fps(p_smiles, tracker, source_file='pool_P_scaffold.parquet', stage='p_fp')
    p_merged = p_merged.iloc[p_valid_idx].reset_index(drop=True)
    
    # Parse P physchem
    p_props = []
    p_scaffs = []
    for i, sm in enumerate(p_merged['smiles']):
        mol = tracker.parse(sm, source_file="pool_P_scaffold.parquet", stage="p_physchem", record_id=str(i))
        if mol:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            p_scaffs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
            from rdkit.Chem import Descriptors
            p_props.append({
                'MW': Descriptors.MolWt(mol),
                'cLogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotB': Descriptors.NumRotatableBonds(mol),
                'FormalCharge': Chem.GetFormalCharge(mol)
            })
        else:
            p_scaffs.append("")
            p_props.append({})
    
    tracker.write_report(os.getcwd())
    p_desc_df = pd.DataFrame(p_props)
    p_merged['murcko'] = p_scaffs
    
    ladders = [('Strict', STRICT), ('R1', R1), ('R2', R2), ('R3', R3), ('R4', R4)]
    
    results = [] # list of (p_id, decoy_id, step)
    used_decoys = set()
    
    print("Beginning Adaptive PMD Search...")
    for step_name, rules in ladders:
        print(f"Applying Level: {step_name}")
        added_in_step = 0
        
        # shuffle P randomly to avoid bias
        indices = np.random.permutation(len(p_merged))
        for i in tqdm(indices):
            p_row = p_desc_df.iloc[i]
            if pd.isna(p_row.get('MW')): continue
            p_scaff = p_merged.iloc[i]['murcko']
            
            # Physchem filter
            mask = get_physchem_mask(p_row, decoy_desc, rules)
            candidate_indices = np.where(mask)[0]
            
            # Remove already used
            candidate_indices = [ci for ci in candidate_indices if ci not in used_decoys]
            
            if not candidate_indices:
                continue
                
            # Compute similarity for candidates against P query
            c_fps = [decoy_fps[ci] for ci in candidate_indices]
            sims = DataStructs.BulkTanimotoSimilarity(p_fps[i], c_fps)
            
            # Find matching similarities that enforce disjoint scaffolds
            valid_c = []
            for j, sim in enumerate(sims):
                if rules['SimMin'] <= sim <= rules['SimMax']:
                    c_idx = candidate_indices[j]
                    if decoy_scaffolds[c_idx] != p_scaff:
                        valid_c.append(c_idx)
                        
            if valid_c:
                # pick a random one
                chosen = np.random.choice(valid_c)
                used_decoys.add(chosen)
                
                results.append({
                    'p_inchikey': p_merged.iloc[i]['ligand_inchikey'],
                    'decoy_ligand_id': decoy_df.iloc[chosen]['ligand_id'],
                    'decoy_inchikey': decoy_df.iloc[chosen]['inchikey'],
                    'decoy_smiles': decoy_df.iloc[chosen]['smiles'],
                    'step': step_name
                })
                added_in_step += 1
                
        print(f"{step_name} added {added_in_step} PMDs. Total = {len(results)}")
        
        if step_name == 'Strict' and len(results) > 0:
            strict_df = pd.DataFrame(results)
            strict_df.to_parquet('data/negatives/pmd_strict.parquet', index=False)
            
        if len(results) >= 5000:
            print("PMD count threshold met. Stopping relaxation.")
            break
            
    # Save finals
    final_df = pd.DataFrame(results) if results else pd.DataFrame()
    if not final_df.empty:
        final_df.to_parquet('data/negatives/pmd_relaxed.parquet', index=False)
    
    # Reports
    with open('docs/phase4E_pmd_generation_report.md', 'w') as f:
        f.write("# Phase 4E Adaptive PMD Generation Report\n\n")
        f.write(f"**Total PMDs Generated:** {len(final_df)}\n\n")
        f.write("## Relaxation Ladder Results\n")
        if not final_df.empty:
            counts = final_df['step'].value_counts()
            for step_n, _ in ladders:
                f.write(f"- {step_n}: {counts.get(step_n, 0)}\n")
        else:
            f.write("No PMDs generated.\n")
            
        if len(final_df) < 1000:
            f.write("\n## Plan B TRIGGERED\n")
            f.write(f"Total PMDs ({len(final_df)}) < 1000 hard floor.\n")
            f.write("Status: PMD DIAGNOSTIC-ONLY\n")
            
            with open('docs/phase4E_planB_decision_packet.md', 'w') as pb:
                pb.write("# Phase 4E Plan B Decision Packet\n\n")
                pb.write(f"Total PMDs generated across all relaxation steps: {len(final_df)}\n")
                pb.write("Reason: Final PMD count is < 1,000, which falls below the hard diagnostic floor to permit sufficiently powered statistics for AUROC testing. We formally declare PMD negatives as diagnostic-only. nnPU metrics will serve as the headline validation.\n")
                

if __name__ == '__main__':
    main()
