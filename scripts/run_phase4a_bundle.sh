#!/bin/bash
# run_phase4a_bundle.sh
# Executes all components that replaced the original phase4a_decoy_audit.py

echo "Starting Phase 4A (Decoy Audit & PMD Construction)..."

# 1. Expand Decoy Pool
echo "Running phase4E_expand_decoy_pool.py..."
python scripts/phase4E_expand_decoy_pool.py
if [ $? -ne 0 ]; then echo "Error expanding decoy pool"; exit 1; fi

# 2. Build PU Pools
echo "Running phase4_build_pu_pools.py..."
python scripts/phase4_build_pu_pools.py
if [ $? -ne 0 ]; then echo "Error building PU pools"; exit 1; fi

# 3. Generate PMD Negatives
echo "Running phase4_generate_pmd_negatives.py..."
python scripts/phase4_generate_pmd_negatives.py
if [ $? -ne 0 ]; then echo "Error generating PMD negatives"; exit 1; fi

# 4. Negative Audit
echo "Running phase4E_negative_audit.py..."
python scripts/phase4E_negative_audit.py
if [ $? -ne 0 ]; then echo "Error running negative audit"; exit 1; fi

echo "Phase 4A Bundle Complete!"
