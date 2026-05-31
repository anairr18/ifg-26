#!/bin/bash
# run_all_phases.sh
# Executes Phase 0 through Phase 4 in sequence.
set -e # Exit immediately if any command fails

echo "Starting Full IFG-26 Pipeline..."

echo "====================================="
echo "        PHASE 0 (Data Pipeline)      "
echo "====================================="
python scripts/download_data.py --dataset all
python scripts/generate_hashes.py
python scripts/raw_schema_inspection.py
python scripts/stereo_audit.py
python scripts/protein_mapping.py

# NOTE: Human review step for protein mapping is normally here.
# Assuming it was completed via Claude and decisions are saved.

echo "====================================="
echo "      PHASE 1 (Dataset Assembly)     "
echo "====================================="
python scripts/endpoint_audit.py
python scripts/_apply_protein_mapping_decisions.py
python scripts/phase1_canonicalize.py
python scripts/phase1_merge_pairs.py

echo "====================================="
echo "       PHASE 2 (Data Splitting)      "
echo "====================================="
python scripts/phase2_split.py
# (Assuming phase2b_cluster_split is embedded or executed sequentially after if present)

echo "====================================="
echo "       PHASE 3 (Featurization)       "
echo "====================================="
python scripts/phase3_featurize.py
python scripts/phase3_baseline_probes.py

echo "====================================="
echo "    PHASE 4 (Diagnostic Overlay)     "
echo "====================================="
echo "[Phase 4A] Building PU Pools..."
python scripts/phase4_build_pu_pools.py

echo "[Phase 4D] Computing Protein Embeddings..."
python scripts/generate_protein_embeddings.py

echo "[Phase 4B] Training nnPU..."
python scripts/phase4_train_nnpu.py

echo "[Phase 4C] Generating PMD Negatives..."
python scripts/phase4_generate_pmd_negatives.py

echo "[Phase 4C] Binary Evaluation..."
python scripts/phase4_binary_eval.py

echo "[Phase 4E] Expanding Decoy Pool..."
python scripts/phase4E_expand_decoy_pool.py

echo "[Phase 4E] Negative Audit..."
python scripts/phase4E_negative_audit.py

echo "[Phase 4E] Generating Strict PMD..."
python scripts/phase4E_generate_strict_pmd.py

echo "[Phase 4E] Similarity Stratified Evaluation..."
python scripts/phase4E_similarity_stratified_eval.py

echo "[Phase 4E] Integrity Freeze..."
python scripts/phase4E_integrity_freeze.py

echo "====================================="
echo "       PHASE 5 (Negative Audit)      "
echo "====================================="
echo "[Phase 5A] Generating Hard-Near Negatives..."
python scripts/generate_hard_negatives.py

echo "[Phase 5B] Running Negative Realism Ladder..."
python scripts/phase5B_negative_ladder.py

echo "====================================="
echo "      ALL PHASES COMPLETE (0-5)      "
echo "====================================="
