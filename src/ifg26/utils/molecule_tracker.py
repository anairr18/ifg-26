import pandas as pd
from rdkit import Chem
from pathlib import Path
import json

class MoleculeTracker:
    def __init__(self, name="global"):
        self.name = name
        self.stats = {
            "total_raw": 0,
            "parsed_ok": 0,
            "empty_string": 0,
            "parse_failure": 0,
            "valence_error": 0,
            "sanitize_failure": 0,
        }
        self.failures = [] # list of dicts
        
    def parse(self, smiles, source_file="unknown", stage="unknown", record_id="unknown"):
        self.stats["total_raw"] += 1
        
        failure_record = {
            "source_file": source_file,
            "script_name": self.name,
            "stage": stage,
            "raw_smiles": smiles,
            "reason": "",
            "record_id": record_id
        }
        
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip() or str(smiles).lower() == "nan":
            self.stats["empty_string"] += 1
            # don't save empty strings to failure csv to save space, unless requested
            return None
            
        params = Chem.SmilesParserParams()
        params.sanitize = False
        mol = Chem.MolFromSmiles(smiles, params)
        
        if mol is None:
            self.stats["parse_failure"] += 1
            failure_record["reason"] = "parse_failure"
            self.failures.append(failure_record)
            return None
            
        # Try full sanitization catching errors
        err = Chem.SanitizeMol(mol, catchErrors=True)
        if err != Chem.SanitizeFlags.SANITIZE_NONE:
            err_str = str(err).upper()
            if "PROPERTIES" in err_str or "VALENCE" in err_str:
                self.stats["valence_error"] += 1
                failure_record["reason"] = "valence_error"
                self.failures.append(failure_record)
                return None
            else:
                self.stats["sanitize_failure"] += 1
                failure_record["reason"] = f"sanitize_failure_{err_str}"
                self.failures.append(failure_record)
                return None
                
        self.stats["parsed_ok"] += 1
        return mol

    def write_report(self, root_dir):
        root = Path(root_dir)
        diag_dir = root / "data" / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        fail_path = diag_dir / f"invalid_molecule_records_{self.name}.csv"
        if self.failures:
            pd.DataFrame(self.failures).to_csv(fail_path, index=False)
            
        stats_path = diag_dir / f"molecule_qc_{self.name}.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
            
        return fail_path, stats_path
