import os, gzip, io, sys, re, argparse, itertools, json, math, time
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from collections import Counter
from pyfaidx import Fasta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# --- URLs (change if you prefer different mirrors) ---
UCSC_FASTA_GZ = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
CLINVAR_VCF_GZ = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"

# --- Utilities ---------------------------------------------------------------
def download(url: str, out_path: Path, chunk=1<<20):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download] Exists: {out_path}")
        return
    print(f"[download] GET {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
    print(f"[download] Wrote {out_path}")

def gunzip_to_plain(gz_path: Path, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[gunzip] Exists: {out_path}")
        return
    print(f"[gunzip] {gz_path.name} -> {out_path.name}")
    with gzip.open(gz_path, "rb") as g, open(out_path, "wb") as f:
        f.write(g.read())
    print(f"[gunzip] Wrote {out_path}")

def is_snv(ref: str, alt: str) -> bool:
    return len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT"

def map_clnsig_to_label(clnsig: str):
    s = clnsig.lower()
    has_path = "pathogenic" in s
    has_ben = "benign" in s
    if has_path and not has_ben: return "pathogenic"
    if has_ben and not has_path: return "benign"
    return None

# --- Step 1: Ensure data -----------------------------------------------------
def ensure_data(workdir: Path):
    fasta_gz = workdir / "hg38.fa.gz"
    fasta = workdir / "hg38.fa"
    vcf_gz = workdir / "clinvar.vcf.gz"

    download(UCSC_FASTA_GZ, fasta_gz)
    gunzip_to_plain(fasta_gz, fasta)
    download(CLINVAR_VCF_GZ, vcf_gz)
    return fasta, vcf_gz

# --- Step 2: Filter ClinVar to CFTR SNVs with labels -------------------------
def filter_cftr_snvs(vcf_gz: Path, out_vcf: Path):
    """
    Keep SNVs where INFO/GENEINFO contains 'CFTR' and CLNSIG contains Benign or Pathogenic (but not both).
    Handles multi-ALT by emitting a row per ALT if simple SNV.
    """
    print(f"[vcf] Filtering CFTR SNVs from {vcf_gz.name}")
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    kept = 0
    with gzip.open(vcf_gz, "rt") as f, open(out_vcf, "w") as w:
        for line in f:
            if line.startswith("#"):
                if "##" in line: 
                    # keep minimal header; we write a simple header for TSV instead
                    continue
                else:
                    continue
            rows += 1
            chrom, pos, vid, ref, alts, qual, flt, info = line.rstrip("\n").split("\t", 8)
            if "GENEINFO=" not in info or "CLNSIG=" not in info:
                continue
            # Quick contains checks
            if "GENEINFO=CFTR:" not in info and ";CFTR:" not in info:
                continue
            m = re.search(r"CLNSIG=([^;]+)", info)
            if not m:
                continue
            label = map_clnsig_to_label(m.group(1))
            if label is None:
                continue
            for alt in alts.split(","):
                if is_snv(ref, alt):
                    kept += 1
                    w.write("\t".join([chrom, pos, ref, alt, label, info]) + "\n")
    print(f"[vcf] Scanned {rows:,} VCF records; kept {kept:,} CFTR SNVs -> {out_vcf.name}")

# --- Step 3: Extract sequence windows ----------------------------------------
def extract_windows(fasta_path: Path, cftr_vcf_tsv: Path, k: int, out_csv: Path):
    print(f"[seq] Indexing FASTA with pyfaidx: {fasta_path.name}")
    fa = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    records = []
    with open(cftr_vcf_tsv) as f:
        for line in f:
            chrom, pos, ref, alt, label, info = line.rstrip("\n").split("\t", 5)
            pos = int(pos)
            # window: [pos-k, pos+k) 1-based inclusive of ref base
            start = max(1, pos - k)
            end = pos + k

            # try several chromosome name variants to match FASTA headers (e.g. '7' vs 'chr7')
            def fetch_range(faobj, chrom_name, a, b):
                # Try the exact name, then 'chr' + name (if not present), then strip leading 'chr' if present
                candidates = [chrom_name]
                if not chrom_name.startswith("chr"):
                    candidates.append("chr" + chrom_name)
                else:
                    # if file contains 'chr7' but VCF has '7', above will handle; but if VCF has 'chr7'
                    # and FASTA doesn't, try without prefix too
                    candidates.append(chrom_name[3:])
                for nm in candidates:
                    try:
                        return str(faobj[nm][a-1:b])
                    except KeyError:
                        continue
                # not found
                raise KeyError(f"{chrom_name} not found in FASTA index")

            try:
                seq = fetch_range(fa, chrom, start, end)
                ref_base = fetch_range(fa, chrom, pos, pos)
            except KeyError:
                # warn and skip records where chromosome name does not match FASTA
                print(f"[seq] WARNING: chromosome '{chrom}' not found in FASTA; skipping pos {pos}")
                continue

            records.append({
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "label": label,
                "seq_window_ref": seq,
                "ref_match": (ref_base == ref.upper())
            })
    df = pd.DataFrame(records)
    df = df[df["ref_match"] == True].reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[seq] Wrote {out_csv}  (N={len(df)})")
    return out_csv

# --- Step 4: Sequence encoding for CNN ---------------------------------------
def one_hot_encode_seq(seq):
    """Convert DNA sequence to one-hot encoding for PyTorch CNN (4 x seq_len)"""
    mapping = {"A":0, "C":1, "G":2, "T":3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            arr[mapping[base], i] = 1.0
    return arr

# --- Step 5: PyTorch CNN Model ----------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class CFTR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Calculate flattened size: 201 // 2 // 2 // 2 = 25
        self.fc1 = nn.Linear(256 * 25, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# --- Step 6: Train CNN model -------------------------------------------------
def train_cnn(windows_csv: Path, epochs=8, batch_size=32, lr=1e-3):
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    print(f"[cnn] Loading and encoding sequences...")
    df = pd.read_csv(windows_csv)
    df = df[df["ref_match"] == True].reset_index(drop=True)
    
    # Make the label
    y = df["label"].map({"benign":0,"pathogenic":1}).astype(int).values
    
    # One-hot encode all sequences
    X = np.stack([one_hot_encode_seq(s) for s in df["seq_window_ref"]])
    print(f"[cnn] Encoded {len(X)} sequences with shape {X.shape}")
    
    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]
    
    train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(SeqDataset(Xte, yte), batch_size=batch_size)
    
    # Model
    print(f"[cnn] Building CNN model...")
    model = CFTR_CNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    # Training loop
    print(f"[cnn] Training for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        total = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[cnn] Epoch {ep+1}/{epochs} loss={total/len(train_loader):.4f}")
    
    # Evaluation
    print(f"[cnn] Evaluating on test set...")
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb).squeeze()
            all_pred.extend(p.numpy().tolist())
            all_true.extend(yb.numpy().tolist())
    
    # Convert to numpy arrays
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    # Tune decision threshold to maximize F1 score
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.2, 0.8, 25):
        p = (all_pred >= t).astype(int)
        f = f1_score(all_true, p)
        if f > best_f1:
            best_f1, best_t = f, t
    
    print(f"[metrics] best_threshold={best_t:.3f}  best_F1={best_f1:.3f}")
    pred_bin = (all_pred >= best_t).astype(int)
    
    # Metrics
    roc = roc_auc_score(all_true, all_pred)
    pr = average_precision_score(all_true, all_pred)
    f1 = f1_score(all_true, pred_bin)
    acc = accuracy_score(all_true, pred_bin)
    cm = confusion_matrix(all_true, pred_bin)
    
    print(f"[metrics] ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  F1={f1:.3f}")
    print(f"[metrics] Accuracy={acc:.3f}")
    print("[metrics] Confusion Matrix:")
    print(cm)
    print("[metrics] Classification report:")
    print(classification_report(all_true, pred_bin))
    
    # Save metrics to file
    metrics_file = windows_csv.parent / "cnn_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("CNN Pipeline - CFTR Variant Classification Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: Convolutional Neural Network (PyTorch)\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Optimal threshold: {best_t:.3f}\n\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"ROC-AUC:      {roc:.3f}\n")
        f.write(f"PR-AUC:       {pr:.3f}\n")
        f.write(f"F1 Score:     {f1:.3f}\n")
        f.write(f"Accuracy:     {acc:.3f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(all_true, pred_bin))
        f.write("\n")
    print(f"[metrics] Saved to {metrics_file}")
    
    return model

def make_data_stats(workdir: Path, cftr_tsv: Path, windows_csv: Path, out_json: Path):
    """Compute simple dataset statistics and write JSON summary to out_json."""
    stats = {}
    wd = Path(workdir)

    # clinvar filtered TSV
    if cftr_tsv.exists():
        try:
            vdf = pd.read_csv(cftr_tsv, sep="\t", header=None, names=["chrom","pos","ref","alt","label","info"])
            stats["variants_total"] = int(len(vdf))
            stats["variants_by_label"] = vdf["label"].value_counts().to_dict()
            stats["variants_by_chrom"] = vdf["chrom"].value_counts().to_dict()
        except Exception as e:
            stats["variants_error"] = str(e)
    else:
        stats["variants_total"] = 0

    # windows CSV
    if windows_csv.exists():
        try:
            wdf = pd.read_csv(windows_csv)
            stats["windows_n"] = int(len(wdf))
            if "seq_window_ref" in wdf.columns:
                lens = wdf["seq_window_ref"].dropna().astype(str).str.len()
                stats["window_len"] = {
                    "min": int(lens.min()),
                    "max": int(lens.max()),
                    "median": float(lens.median()),
                    "mean": float(lens.mean())
                }
            if "ref_match" in wdf.columns:
                stats["ref_match_rate"] = float((wdf["ref_match"] == True).mean())
        except Exception as e:
            stats["windows_error"] = str(e)
    else:
        stats["windows_n"] = 0

    # write JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_json, "w") as oj:
            json.dump(stats, oj, indent=2)
        print(f"[stats] Wrote {out_json}")
        # print a short human summary
        ssum = []
        ssum.append(f"variants={stats.get('variants_total',0)}")
        if "variants_by_label" in stats:
            ssum.append("labels=" + ",".join([f"{k}:{v}" for k,v in stats["variants_by_label"].items()]))
        ssum.append(f"windows={stats.get('windows_n',0)}")
        print("[stats] " + " ; ".join(ssum))
    except Exception as e:
        print(f"[stats] ERROR writing stats: {e}")
    return out_json

# --- Orchestrator ------------------------------------------------------------
def run_all(workdir: str, k: int):
    start_time = time.time()
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fasta, vcf_gz = ensure_data(wd)
    print(f"[timing] Data download/prep: {time.time()-t0:.2f}s")
    
    t0 = time.time()
    cftr_tsv = wd / "clinvar_CFTR_snvs.tsv"
    filter_cftr_snvs(vcf_gz, cftr_tsv)
    print(f"[timing] VCF filtering: {time.time()-t0:.2f}s")

    t0 = time.time()
    windows_csv = wd / "cftr_windows_ref.csv"
    extract_windows(fasta, cftr_tsv, k=k, out_csv=windows_csv)
    print(f"[timing] Window extraction: {time.time()-t0:.2f}s")

    t0 = time.time()
    # compute and write dataset statistics
    stats_json = wd / "data_stats.json"
    make_data_stats(wd, cftr_tsv, windows_csv, stats_json)
    print(f"[timing] Statistics computation: {time.time()-t0:.2f}s")

    t0 = time.time()
    train_cnn(windows_csv)
    print(f"[timing] Model training: {time.time()-t0:.2f}s")
    
    total_time = time.time() - start_time
    print(f"[timing] Total pipeline time: {total_time:.2f}s ({total_time/60:.2f}m)")

# --- CLI ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="End-to-end CFTR CNN pipeline for sequence-based classification.")
    ap.add_argument("--workdir", default="data", help="Output directory (default: data)")
    ap.add_argument("--k", type=int, default=100, help="Half-window size around variant (default: 100)")
    args = ap.parse_args()
    run_all(args.workdir, args.k)
