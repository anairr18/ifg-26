import subprocess
import os
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"ERROR: Command failed with exit code {res.returncode}")
        sys.exit(1)
    print("SUCCESS\n")

if __name__ == "__main__":
    print("==========================================")
    print(" Running IFG-26 MVP Python Smoke Tests")
    print("==========================================")
    
    os.makedirs("data", exist_ok=True)
    with open("data/dummy_triplets.csv", "w") as f:
        f.write("ligand_smiles,e3_protein,neosubstrate,label,source\n")
        f.write("C1=CC=CC=C1,CRBN,IKZF1,1,MGDB\n")
        f.write("C1=CC=CC=C1C,CRBN,GSPT1,1,MGTbind\n")
        f.write("C1=CC=CC=C1CC,VHL,IKZF1,1,MGDB\n")
        f.write("C1=CC=CC=C1CCC,VHL,GSPT1,1,MGTbind\n")
        f.write("C1=CC=CC=C1CCCC,CRBN,IKZF1,1,MGDB\n")
        
    with open("data/dummy_decoys.csv", "w") as f:
        f.write("ligand_smiles,e3_protein,neosubstrate,label,source\n")
        f.write("CCO,CRBN,IKZF1,0,ChEMBL\n")
        f.write("CCC,CRBN,GSPT1,0,ChEMBL\n")
        f.write("CCCC,VHL,IKZF1,0,ChEMBL\n")
        f.write("CCCCC,VHL,GSPT1,0,ChEMBL\n")
        
    print("[1/4] Running model list check...")
    run_cmd("python run_ifg26_benchmark.py --list-models")
    
    print("[2/4] Running dry-run...")
    run_cmd("python run_ifg26_benchmark.py --triplets data/dummy_triplets.csv --decoys data/dummy_decoys.csv --dry-run --outdir /tmp/ifg26_smoke")
    
    print("[3/4] Running baseline random split...")
    run_cmd("python run_ifg26_benchmark.py --triplets data/dummy_triplets.csv --decoys data/dummy_decoys.csv --model rf --split random --seed 42 --outdir /tmp/ifg26_smoke")
    
    print("[4/4] Running bootstrap evaluation on scaffold split with diagnostics...")
    run_cmd("python run_ifg26_benchmark.py --triplets data/dummy_triplets.csv --decoys data/dummy_decoys.csv --model rf --split scaffold --bootstrap 50 --seed 42 --diagnostics --outdir /tmp/ifg26_smoke")

    print("[+] Running Mechanism Tests...")
    run_cmd("python mechanistic_tests/interface_perturbation.py")

    required = [
        "/tmp/ifg26_smoke/results/metrics.json",
        "/tmp/ifg26_smoke/results/bootstrap_metrics.json",
        "/tmp/ifg26_smoke/results/shortcut_diagnostics.json",
        "/tmp/ifg26_smoke/results/split_manifest.json",
        "/tmp/ifg26_smoke/results/experiment_metadata.json"
    ]
    
    for req in required:
        if not os.path.exists(req):
            print(f"FAILED: Missing {req}")
            sys.exit(1)
            
    print("All Python Smoke Tests Passed!")
