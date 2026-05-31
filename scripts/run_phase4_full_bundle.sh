#!/bin/bash
# run_phase4_full_bundle.sh
# Executes all Phase 4 Diagnostics in sequence.
set -e # Exit immediately if any command fails

echo "Starting Phase 4 (Diagnostic Overlay)..."

echo "[Phase 4A] Building PU Pools..."
python scripts/phase4_build_pu_pools.py

echo "[Phase 4D] Computing Protein Embeddings..."
python scripts/phase4_compute_protein_embeddings.py

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

echo "Phase 4 Execution Complete!"
