#!/bin/bash
set -e

echo "=========================================="
echo " Running IFG-26 MVP Smoke Tests"
echo "=========================================="

# Create dummy data for tests
mkdir -p data
cat <<EOF > data/dummy_triplets.csv
ligand_smiles,e3_protein,neosubstrate,label,source
CC(=O)Oc1ccccc1C(=O)O,CRBN,IKZF1,1,MGDB
c1ccccc1,CRBN,GSPT1,1,MGTbind
CC(=O)Oc1ccccc1C(=O)O,VHL,IKZF1,1,MGDB
c1ccccc1,VHL,GSPT1,1,MGTbind
EOF

cat <<EOF > data/dummy_decoys.csv
ligand_smiles,e3_protein,neosubstrate,label,source
CCO,CRBN,IKZF1,0,ChEMBL
CCC,CRBN,GSPT1,0,ChEMBL
EOF

echo "[1/4] Running model list check..."
python run_ifg26_benchmark.py --list-models

echo "[2/4] Running dry-run..."
python run_ifg26_benchmark.py --triplets data/dummy_triplets.csv --decoys data/dummy_decoys.csv --dry-run --outdir /tmp/ifg26_smoke

echo "[3/4] Running baseline random split..."
python run_ifg26_benchmark.py --triplets data/dummy_triplets.csv --decoys data/dummy_decoys.csv --model rf --split random --seed 42 --outdir /tmp/ifg26_smoke

echo "[4/4] Running bootstrap evaluation on scaffold split with diagnostics..."
python run_ifg26_benchmark.py --triplets data/dummy_triplets.csv --decoys data/dummy_decoys.csv --model rf --split scaffold --bootstrap 50 --seed 42 --diagnostics --outdir /tmp/ifg26_smoke

# Verify output files exist
if [ ! -f "/tmp/ifg26_smoke/results/metrics.json" ]; then
    echo "ERROR: metrics.json missing!"
    exit 1
fi

if [ ! -f "/tmp/ifg26_smoke/results/bootstrap_metrics.json" ]; then
    echo "ERROR: bootstrap_metrics.json missing!"
    exit 1
fi

if [ ! -f "/tmp/ifg26_smoke/results/shortcut_diagnostics.json" ]; then
    echo "ERROR: shortcut_diagnostics.json missing!"
    exit 1
fi

if [ ! -f "/tmp/ifg26_smoke/results/split_manifest.json" ]; then
    echo "ERROR: split_manifest.json missing!"
    exit 1
fi

if [ ! -f "/tmp/ifg26_smoke/results/experiment_metadata.json" ]; then
    echo "ERROR: experiment_metadata.json missing!"
    exit 1
fi

python mechanistic_tests/interface_perturbation.py

echo "=========================================="
echo " All Smoke Tests Passed!"
echo "=========================================="
