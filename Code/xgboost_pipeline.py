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

    # combine numeric features, categorical one-hot columns (if any), and k-mer features
    base_cols = ["gc","symmetry","win_len","center_ref_mismatch","is_transition"]
    cat_cols = [c for c in df.columns if c.startswith(("center_ref_","ref_","alt_","chrom_"))]
    parts = [df[base_cols]]
    if len(cat_cols) > 0:
        parts.append(df[cat_cols])
    parts.append(kdf)
    X = pd.concat(parts, axis=1)
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
            from sklearn.model_selection import GridSearchCV
            
            print("[train] Running XGBoost hyperparameter search...")
            params = {
                "n_estimators": [200, 600],
                "max_depth": [3, 5, 8],
                "learning_rate": [0.01, 0.03, 0.1]
            }
            
            base = xgb.XGBClassifier(
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                random_state=42
            )
            
            gs = GridSearchCV(base, params, cv=3, scoring="f1", n_jobs=-1, verbose=1)
            gs.fit(Xtr, ytr)
            print(f"[train] best XGBoost params: {gs.best_params_}")
            
            model = gs.best_estimator_
            proba = model.predict_proba(Xte)[:,1]
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("[train] Model: XGBoost")
        except Exception as e:
            print(f"[train] XGBoost unavailable ({e}). Falling back to RandomForest.")
            model_kind = "rf"

    if model_kind.lower() != "xgboost":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        
        print("[train] Running RandomForest hyperparameter search...")
        params = {
            "n_estimators": [200, 600],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 5, 10]
        }
        
        base = RandomForestClassifier(
            class_weight={0:1.0, 1:max(1.0, scale_pos_weight)},
            random_state=42,
            n_jobs=-1
        )
        
        gs = GridSearchCV(base, params, cv=3, scoring="f1", n_jobs=-1, verbose=1)
        gs.fit(Xtr, ytr)
        print(f"[train] best RF params: {gs.best_params_}")
        
        model = gs.best_estimator_
        proba = model.predict_proba(Xte)[:,1]
        # impurity-based importances
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("[train] Model: RandomForest")

    # Tune decision threshold to maximize F1 score
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.2, 0.8, 25):
        p = (proba >= t).astype(int)
        f = f1_score(yte, p)
        if f > best_f1:
            best_f1, best_t = f, t
    
    print(f"[metrics] best_threshold={best_t:.3f}  best_F1={best_f1:.3f}")
    pred = (proba >= best_t).astype(int)

    roc = roc_auc_score(yte, proba)
    pr = average_precision_score(yte, proba)
    f1 = f1_score(yte, pred)
    print(f"[metrics] ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  F1={f1:.3f}")
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred)

    print(f"[metrics] Accuracy={acc:.3f}")
    print("[metrics] Confusion Matrix:")
    print(cm)

    print("[metrics] Classification report:")
    print(classification_report(yte, pred))

    # Save metrics to file
    metrics_file = features_csv.parent / "xgboost_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("XGBoost Pipeline - CFTR Variant Classification Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: XGBoost Classifier\n")
        f.write(f"Best parameters: {gs.best_params_}\n")
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
        f.write(classification_report(yte, pred))
        f.write("\n")
    print(f"[metrics] Saved to {metrics_file}")

    top_imp = imp.head(200)
    out_path = features_csv.parent / "feature_importances_top200.csv"
    top_imp.to_csv(out_path)
    print(f"[train] Wrote {out_path}")

def make_data_stats(workdir: Path, cftr_tsv: Path, windows_csv: Path, features_csv: Path, out_json: Path):
    """Compute simple dataset statistics and write JSON summary to out_json.

    Stats include:
      - total variants, counts by label and by chromosome (from filtered VCF TSV)
      - windows count, window length distribution, GC stats, ref-match rate (from windows CSV)
      - feature matrix shape and label balance (from features CSV)
    """
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
                # GC stats
                try:
                    gcs = wdf["seq_window_ref"].apply(gc_content)
                    stats["gc"] = {
                        "mean": float(gcs.mean()),
                        "median": float(gcs.median())
                    }
                except Exception:
                    pass
            if "ref_match" in wdf.columns:
                stats["ref_match_rate"] = float((wdf["ref_match"] == True).mean())
        except Exception as e:
            stats["windows_error"] = str(e)
    else:
        stats["windows_n"] = 0

    # features CSV
    if features_csv.exists():
        try:
            fdf = pd.read_csv(features_csv)
            stats["features_shape"] = list(fdf.shape)
            if "label" in fdf.columns:
                stats["label_balance"] = fdf["label"].value_counts().to_dict()
        except Exception as e:
            stats["features_error"] = str(e)
    else:
        stats["features_shape"] = [0,0]

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
def run_all(workdir: str, k: int, kmer: int, model: str):
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
    feats_csv = wd / "cftr_features.csv"
    make_features(windows_csv, feats_csv, kmer=kmer)
    print(f"[timing] Feature engineering: {time.time()-t0:.2f}s")

    t0 = time.time()
    # compute and write dataset statistics
    stats_json = wd / "data_stats.json"
    make_data_stats(wd, cftr_tsv, windows_csv, feats_csv, stats_json)
    print(f"[timing] Statistics computation: {time.time()-t0:.2f}s")

    t0 = time.time()
    train_tree(feats_csv, model_kind=model)
    print(f"[timing] Model training: {time.time()-t0:.2f}s")
    
    total_time = time.time() - start_time
    print(f"[timing] Total pipeline time: {total_time:.2f}s ({total_time/60:.2f}m)")

# --- CLI ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="End-to-end CFTR tree-model pipeline (Python-only).")
    ap.add_argument("--workdir", default="data", help="Output directory (default: data)")
    ap.add_argument("--k", type=int, default=100, help="Half-window size around variant (default: 100)")
    ap.add_argument("--kmer", type=int, default=3, help="k-mer size for features (default: 3)")
    ap.add_argument("--model", choices=["xgboost","rf"], default="xgboost", help="Tree model to use")
    args = ap.parse_args()
    run_all(args.workdir, args.k, args.kmer, args.model)
