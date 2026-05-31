"""
phase4_compute_protein_embeddings.py
=====================================
IFG-26 Phase 4D — ESM2 Protein Embeddings.

Fetches amino acid sequences from UniProt REST API for all unique proteins
in the scaffold split and encodes them with ESM2-8M (frozen, CPU).

Replaces Phase 3 Tier-P2 stub with real 320-dim embeddings.

Model: esm2_t6_8M_UR50D (6 layers, embed_dim=320, fast on CPU)

Outputs:
    data/features/protein_embeddings_esm2.parquet
        columns: uniprot_id, embedding (320-float list), seq_len,
                 esm2_model, fetch_ok, error_note
    docs/phase4D_protein_shortcut_diagnosis.md

Also updates data/features/protein_index.parquet with esm2_available flag.
"""

import argparse
import json
import logging
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore")

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "phase4_default.yaml"


def setup_logging(name="phase4_esm2"):
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); lg.addHandler(sh)
    return lg


def fetch_sequence(uniprot_id: str, timeout: int = 10) -> str | None:
    """Fetch canonical FASTA sequence from UniProt REST API."""
    # Handle composite IDs like P01116_P15056 → use first
    uid = uniprot_id.split("_")[0].strip()
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            fasta = resp.read().decode("utf-8")
        lines = fasta.strip().split("\n")
        seq = "".join(lines[1:])  # skip header
        return seq if seq else None
    except Exception as e:
        return None


def embed_sequences(model, batch_converter, seqs: list[tuple[str, str]],
                    repr_layer: int, device: torch.device,
                    batch_size: int) -> dict[str, np.ndarray]:
    """Run ESM2 on a list of (name, seq) tuples. Returns {name: embedding}."""
    results = {}
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        labels, strs, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        for j, (name, _) in enumerate(batch):
            # Mean over sequence positions (excluding BOS/EOS)
            seq_len = tokens[j].ne(model.padding_idx).sum().item() - 2  # minus BOS+EOS
            repr_vec = out["representations"][repr_layer][j, 1:seq_len+1].mean(0)
            results[name] = repr_vec.cpu().numpy()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lg = setup_logging()
    ts = datetime.now(timezone.utc).isoformat()
    lg.info("=" * 70)
    lg.info(f"IFG-26 Phase 4D — ESM2 Protein Embeddings  {ts}")
    lg.info("=" * 70)

    if not ESM_AVAILABLE:
        lg.error("fair-esm not installed. Run: pip install fair-esm")
        sys.exit(1)

    emb_cfg  = cfg.get("protein_embeddings", {})
    model_name = emb_cfg.get("model", "esm2_t6_8M_UR50D")
    repr_layer = emb_cfg.get("repr_layer", 6)
    batch_size = emb_cfg.get("batch_size", 8)
    timeout    = emb_cfg.get("timeout_per_seq", 10)
    cache_path = ROOT / emb_cfg.get("cache_path", "data/features/protein_embeddings_esm2.parquet")
    features_dir = ROOT / cfg["outputs"]["features_dir"]

    if cache_path.exists():
        lg.info(f"Cache exists at {cache_path.name} — loading and skipping re-embedding.")
        emb_df = pd.read_parquet(cache_path)
        if emb_df["fetch_ok"].all():
            lg.info(f"  All {len(emb_df)} proteins already embedded. Done.")
            _write_diagnosis(emb_df, cfg, lg, ts)
            return
        lg.info(f"  Partial cache: {emb_df['fetch_ok'].sum()} ok, "
                f"{(~emb_df['fetch_ok']).sum()} failed. Re-trying failed proteins.")
        done_ids = set(emb_df[emb_df["fetch_ok"]]["uniprot_id"])
        p_idx    = pd.read_parquet(features_dir / "protein_index.parquet")
        all_ids  = [u for u in p_idx["uniprot_id"].tolist() if u not in done_ids]
    else:
        done_ids = set()
        p_idx    = pd.read_parquet(features_dir / "protein_index.parquet")
        all_ids  = p_idx["uniprot_id"].tolist()

    lg.info(f"Unique UniProt IDs to embed: {len(all_ids)}")

    # ── Load ESM2 ─────────────────────────────────────────────────────
    lg.info(f"Loading {model_name}...")
    loader_fn = getattr(esm.pretrained, model_name, None)
    if loader_fn is None:
        abort_msg = f"Unknown ESM2 model: {model_name}. Check esm.pretrained.*"
        lg.error(abort_msg); sys.exit(1)
    esm_model, alphabet = loader_fn()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cpu")
    esm_model = esm_model.to(device).eval()
    embed_dim = esm_model.embed_dim
    lg.info(f"  Loaded {model_name}: embed_dim={embed_dim}, repr_layer={repr_layer}")

    # ── Fetch sequences from UniProt ──────────────────────────────────
    lg.info("Fetching sequences from UniProt REST API...")
    id_to_seq: dict[str, str | None] = {}
    for i, uid in enumerate(all_ids):
        seq = fetch_sequence(uid, timeout)
        id_to_seq[uid] = seq
        if seq:
            if (i + 1) % 20 == 0:
                lg.info(f"  {i+1}/{len(all_ids)} fetched | latest: {uid} ({len(seq)} aa)")
        else:
            lg.warning(f"  Fetch FAILED: {uid}")
        time.sleep(0.1)  # polite rate limit

    n_ok   = sum(1 for v in id_to_seq.values() if v)
    n_fail = sum(1 for v in id_to_seq.values() if not v)
    lg.info(f"Sequence fetch: {n_ok} OK, {n_fail} failed")

    # ── Embed with ESM2 ───────────────────────────────────────────────
    lg.info("Computing ESM2 embeddings...")
    seq_tuples = [(uid, seq) for uid, seq in id_to_seq.items() if seq]
    # Truncate very long sequences (ESM2-8M handles up to ~1024 tokens safely on CPU)
    seq_tuples = [(uid, seq[:1022]) for uid, seq in seq_tuples]

    embeddings = embed_sequences(esm_model, batch_converter, seq_tuples,
                                 repr_layer, device, batch_size)
    lg.info(f"  Embedded {len(embeddings)} proteins")

    # ── Build output dataframe ─────────────────────────────────────────
    rows = []
    for uid in all_ids:
        seq = id_to_seq.get(uid)
        emb = embeddings.get(uid)
        rows.append({
            "uniprot_id": uid,
            "esm2_model": model_name,
            "embed_dim":  embed_dim,
            "seq_len":    len(seq) if seq else 0,
            "fetch_ok":   bool(seq and emb is not None),
            "error_note": "" if (seq and emb is not None) else
                          ("fetch_failed" if not seq else "embed_failed"),
            "embedding":  emb.tolist() if emb is not None else [0.0] * embed_dim,
        })

    new_df = pd.DataFrame(rows)

    # Merge with any previously cached successful rows
    if done_ids:
        old_ok = emb_df[emb_df["fetch_ok"]]
        emb_df = pd.concat([old_ok, new_df], ignore_index=True)
    else:
        emb_df = new_df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(cache_path, index=False)
    lg.info(f"\n  Written: {cache_path.name} ({len(emb_df)} proteins, "
            f"{emb_df['fetch_ok'].sum()} with valid embeddings)")

    # Update protein_index.parquet with esm2_available flag
    p_idx_updated = p_idx.copy()
    ok_set = set(emb_df[emb_df["fetch_ok"]]["uniprot_id"])
    p_idx_updated["esm2_available"] = p_idx_updated["uniprot_id"].isin(ok_set)
    p_idx_updated.to_parquet(features_dir / "protein_index.parquet", index=False)
    lg.info(f"  Updated protein_index.parquet with esm2_available flag")

    _write_diagnosis(emb_df, cfg, lg, ts)


def _write_diagnosis(emb_df: pd.DataFrame, cfg: dict,
                     lg: logging.Logger, ts: str):
    """Write phase4D_protein_shortcut_diagnosis.md."""
    n_ok   = int(emb_df["fetch_ok"].sum())
    n_fail = int((~emb_df["fetch_ok"]).sum())
    model  = cfg.get("protein_embeddings", {}).get("model", "esm2_t6_8M_UR50D")

    # Probe shortcut assessment note (actual metric filled by phase4_train_nnpu.py)
    diag_md = f"""# IFG-26 Phase 4D — Protein Shortcut Diagnosis

_Generated: {ts}_

## ESM2 Embedding Summary

| Item | Value |
|---|---|
| Model | {model} |
| Embedding dim | {emb_df['embed_dim'].iloc[0] if len(emb_df) else '?'} |
| Proteins embedded | {n_ok} |
| Fetch failures | {n_fail} |
| Failures go to | zero-vector (excluded from LP0 training) |

## Protein Shortcut Criterion

> **If LP0 (ligand+protein) outperforms L0 (ligand-only) by > 5pp in Bin A
> (NN < 0.60), this indicates protein identity is driving generalisation**,
> not chemical structure recognition.

Bin A is the strictest OOD regime: test ligands have max Tanimoto < 0.60
to any train ligand. A model truly learning chemical structure should perform
similarly on L0 and LP0 in this bin.

| Model | Bin-A Recall@5% | Diagnosis |
|---|---|---|
| nnPU-L0  | _(filled by Phase 4B)_ | — |
| nnPU-LP0 | _(filled by Phase 4B)_ | — |

If `LP0_Bin_A - L0_Bin_A > 5pp`:
- ⚠️ Protein identity shortcut suspected
- Recommended remediation: homology-split held-out evaluation (Phase 5)

## Failed Proteins

```
{chr(10).join(emb_df[~emb_df["fetch_ok"]]["uniprot_id"].tolist()) or "None"}
```
"""
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    with open(docs_dir / "phase4D_protein_shortcut_diagnosis.md", "w", encoding="utf-8") as f:
        f.write(diag_md)
    lg.info("  Written: docs/phase4D_protein_shortcut_diagnosis.md")
    lg.info("\n" + "=" * 70)
    lg.info("IFG-26 Phase 4D — COMPLETE")
    lg.info("=" * 70)


if __name__ == "__main__":
    main()
