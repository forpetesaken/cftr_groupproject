import os, gzip, io, sys, re, argparse, itertools, json, math
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
            seq = str(fa[chrom][start-1:end])  # pyfaidx uses 0-based slices
            ref_base = str(fa[chrom][pos-1:pos])

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

# --- Step 4: Feature engineering ---------------------------------------------
NUCS = ["A","C","G","T"]

def gc_content(seq):
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    atgc = sum(s.count(b) for b in NUCS)
    return gc / max(1, atgc)

def revcomp(seq):
    comp = str.maketrans("ACGTacgt","TGCAtgca")
    return seq.translate(comp)[::-1]

def symmetry(seq):
    s, rc = seq.upper(), revcomp(seq.upper())
    n = min(len(s), len(rc))
    return 0.0 if n==0 else float(np.mean([s[i]==rc[i] for i in range(n)]))

def is_transition(r,a):
    pur, pyr = set(["A","G"]), set(["C","T"])
    r, a = r.upper(), a.upper()
    return int((r in pur and a in pur) or (r in pyr and a in pyr))

def kmer_vec(seq, k, vocab):
    s = re.sub(r"[^ACGT]", "N", seq.upper())
    cnt = Counter(s[i:i+k] for i in range(len(s)-k+1))
    v = np.array([cnt.get(km,0) for km in vocab], dtype=np.float32)
    tot = v.sum() or 1.0
    return v / tot

def make_features(windows_csv: Path, out_csv: Path, kmer=3):
    print(f"[feat] Reading {windows_csv}")
    df = pd.read_csv(windows_csv)
    df = df[df["ref_match"]==True].copy()

    # center base (assumes windows are centered)
    L = int(df["seq_window_ref"].str.len().median())
    center = max(0, L//2)
    def center_base(s):
        return s[center].upper() if isinstance(s, str) and len(s)>center else "N"

    df["gc"] = df["seq_window_ref"].apply(gc_content)
    df["symmetry"] = df["seq_window_ref"].apply(symmetry)
    df["win_len"] = df["seq_window_ref"].str.len()
    df["center_ref"] = df["seq_window_ref"].apply(center_base)
    df["center_ref_mismatch"] = (df["center_ref"] != df["ref"].str.upper()).astype(int)
    df["is_transition"] = [is_transition(r,a) for r,a in zip(df["ref"], df["alt"])]

    # one-hots for small categoricals
    for col in ["center_ref","ref","alt","chrom"]:
        df = pd.concat([df, pd.get_dummies(df[col].astype(str), prefix=col)], axis=1)

    # k-mer features
    vocab = ["".join(p) for p in itertools.product(NUCS, repeat=kmer)]
    km = np.vstack(df["seq_window_ref"].apply(lambda s: kmer_vec(s, kmer, vocab)).values)
    kcols = [f"kmer_{k}" for k in vocab]
    kdf = pd.DataFrame(km, columns=kcols, index=df.index)

    X = pd.concat([df[["gc","symmetry","win_len","center_ref_mismatch","is_transition"]]
                  + [c for c in df.columns if c.startswith(("center_ref_","ref_","alt_","chrom_"))],
                   kdf], axis=1)
    y = df["label"].map({"benign":0,"pathogenic":1}).astype(int)

    out = pd.concat([X, y.rename("label")], axis=1)
    out.to_csv(out_csv, index=False)
    print(f"[feat] Wrote {out_csv}  shape={out.shape}")
    return out_csv

# --- Step 5: Train a tree model ----------------------------------------------
def train_tree(features_csv: Path, model_kind: str = "xgboost"):
    df = pd.read_csv(features_csv)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pos = int(ytr.sum()); neg = len(ytr) - pos
    scale_pos_weight = neg / max(1, pos)

    if model_kind.lower() == "xgboost":
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=800,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                random_state=42
            )
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:,1]
            pred = (proba >= 0.5).astype(int)
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("[train] Model: XGBoost")
        except Exception as e:
            print(f"[train] XGBoost unavailable ({e}). Falling back to RandomForest.")
            model_kind = "rf"

    if model_kind.lower() != "xgboost":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            class_weight={0:1.0, 1:max(1.0, scale_pos_weight)},
            random_state=42,
            n_jobs=-1
        )
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:,1]
        pred = (proba >= 0.5).astype(int)
        # impurity-based importances
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("[train] Model: RandomForest")

    roc = roc_auc_score(yte, proba)
    pr = average_precision_score(yte, proba)
    f1 = f1_score(yte, pred)
    print(f"[metrics] ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  F1={f1:.3f}")

    top_imp = imp.head(200)
    out_path = features_csv.parent / "feature_importances_top200.csv"
    top_imp.to_csv(out_path)
    print(f"[train] Wrote {out_path}")

# --- Orchestrator ------------------------------------------------------------
def run_all(workdir: str, k: int, kmer: int, model: str):
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    fasta, vcf_gz = ensure_data(wd)
    cftr_tsv = wd / "clinvar_CFTR_snvs.tsv"
    filter_cftr_snvs(vcf_gz, cftr_tsv)

    windows_csv = wd / "cftr_windows_ref.csv"
    extract_windows(fasta, cftr_tsv, k=k, out_csv=windows_csv)

    feats_csv = wd / "cftr_features.csv"
    make_features(windows_csv, feats_csv, kmer=kmer)

    train_tree(feats_csv, model_kind=model)

# --- CLI ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="End-to-end CFTR tree-model pipeline (Python-only).")
    ap.add_argument("--workdir", default="data", help="Output directory (default: data)")
    ap.add_argument("--k", type=int, default=100, help="Half-window size around variant (default: 100)")
    ap.add_argument("--kmer", type=int, default=3, help="k-mer size for features (default: 3)")
    ap.add_argument("--model", choices=["xgboost","rf"], default="xgboost", help="Tree model to use")
    args = ap.parse_args()
    run_all(args.workdir, args.k, args.kmer, args.model)
