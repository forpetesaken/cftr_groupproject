# Data Description and Preparation

## Data Sources and Relevance

### Primary Data Sources

#### 1. ClinVar Variant Database
- **Source**: NCBI ClinVar VCF (GRCh38) - `ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz`
- **Size**: 4,124,089 total variant records
- **Relevance**: ClinVar is a freely accessible public archive of reports on the relationships among human variations and phenotypes. It provides expert-reviewed clinical interpretations of genetic variants, making it the gold standard for variant pathogenicity classification.
- **Gene Focus**: CFTR (Cystic Fibrosis Transmembrane Conductance Regulator) gene on chromosome 7
- **Clinical Significance**: Contains variants classified as "Benign" or "Pathogenic" by clinical experts
- **Filtered Dataset**: 2,544 CFTR single nucleotide variants (SNVs)
  - 1,448 Benign variants (56.9%)
  - 1,096 Pathogenic variants (43.1%)

#### 2. Human Reference Genome (hg38)
- **Source**: UCSC Genome Browser - `hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz`
- **Assembly**: GRCh38/hg38 (latest human reference genome)
- **Size**: ~3 GB compressed FASTA file
- **Relevance**: Provides the reference DNA sequences necessary to extract genomic context around each variant. Sequence context is critical for understanding how variants affect protein structure and function.
- **Usage**: Extract 200bp windows (±100bp) around each variant position to capture local sequence features

### Data Type Classification
- **Omics Data Type**: Genomics (DNA sequence data)
- **Application Domain**: Clinical variant interpretation and pathogenicity prediction
- **Machine Learning Task**: Binary classification (Benign vs. Pathogenic)

---

## Data Preprocessing Pipeline

### 1. Variant Filtering and Quality Control

#### Initial Filtering Criteria
```python
# Filter variants from 4.1M records to 2,544 CFTR SNVs
- Gene: Must contain GENEINFO=CFTR in VCF INFO field
- Variant Type: Single nucleotide variants (SNVs) only
  - Reference: Single nucleotide (A, C, G, T)
  - Alternate: Single nucleotide (A, C, G, T)
  - Excludes: Insertions, deletions, multi-nucleotide variants
- Clinical Significance: Clear pathogenic or benign classification
  - "Pathogenic" → Label: 1
  - "Benign" → Label: 0
  - Excluded: Conflicting interpretations, VUS (Variants of Uncertain Significance)
```

#### Quality Assurance
- **Reference Allele Validation**: Cross-check VCF reference allele against hg38 FASTA
- **Chromosome Name Matching**: Handle both '7' and 'chr7' formats between VCF and FASTA
- **Multi-allelic Site Handling**: Split multi-allelic variants into separate records

**Output**: `clinvar_CFTR_snvs.tsv` (2,544 filtered variants)

---

### 2. Sequence Window Extraction

#### Window Definition
- **Window Size**: 200 base pairs (k=100 on each side of variant)
- **Center Position**: Variant position from ClinVar
- **Coordinates**: 1-based inclusive ([pos-100, pos+100])
- **Boundary Handling**: Max(1, pos-k) to avoid negative coordinates

#### Reference Matching
```python
# Validation step
ref_base = fasta.fetch(chrom, pos, pos)
ref_match = (ref_base == vcf_ref)  # Must be TRUE
```
- Only windows with `ref_match=True` are retained
- Ensures alignment between VCF and reference genome

**Output**: `cftr_windows_ref.csv` (2,544 sequences × 200bp)

---

### 3. Feature Engineering

#### A. Sequence Composition Features (5 features)

1. **GC Content** (1 feature)
   ```python
   gc_content = (count_G + count_C) / (count_A + count_T + count_G + count_C)
   ```
   - Range: [0, 1]
   - Biological significance: GC-rich regions have different stability and mutation rates

2. **Sequence Symmetry** (1 feature)
   ```python
   symmetry = mean(seq[i] == reverse_complement(seq)[i] for i in range(len))
   ```
   - Range: [0, 1]
   - Captures palindromic structures and DNA secondary structure potential

3. **Window Length** (1 feature)
   - Actual sequence length retrieved
   - Accounts for boundary effects near chromosome ends

4. **Center Reference Base** (1 feature)
   - Nucleotide at variant position
   - Encoded as categorical (A, C, G, T)

5. **Center Reference Mismatch** (1 feature)
   - Binary indicator of VCF/FASTA discordance
   - Quality control flag

#### B. Variant-Specific Features (1 feature)

**Transition vs. Transversion** (1 feature)
```python
Transitions: A↔G, C↔T (purines ↔ purines, pyrimidines ↔ pyrimidines)
Transversions: A↔C, A↔T, G↔C, G↔T (purine ↔ pyrimidine)
is_transition = 1 if transition, 0 if transversion
```
- **Biological Relevance**: Transitions are ~2× more common than transversions in human genomes
- Affects mutational spectrum and pathogenicity likelihood

#### C. K-mer Frequency Features (64 features)

**3-mer (Trinucleotide) Vocabulary**
```python
vocab = ['AAA', 'AAC', 'AAG', 'AAT', ..., 'TTT']  # 4^3 = 64 k-mers
```

**Normalized Frequency Vectors**
```python
for each 3-mer:
    count = occurrences in 200bp window
    frequency = count / total_kmers
```

- **Biological Significance**: Trinucleotide context affects:
  - DNA methylation patterns (CpG dinucleotides)
  - Mutational signatures
  - Codon usage bias
  - Regulatory motifs

#### D. One-Hot Encoded Categorical Features (14 features)

1. **Center Reference Base** (4 features): A, C, G, T
2. **Reference Allele** (4 features): A, C, G, T
3. **Alternate Allele** (4 features): A, C, G, T
4. **Chromosome** (2 features): chr7, 7 (handles both naming conventions)

**Total Engineered Features**: 5 + 1 + 64 + 14 = **84 features** + label column

**Output**: `cftr_features.csv` (2,544 samples × 85 columns)

---

### 4. Data Normalization and Scaling

#### For Tree-Based Models (XGBoost, LightGBM)
- **No scaling applied**
- Tree-based models are invariant to monotonic transformations
- Features used in original scale

#### For SVM Model
```python
# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
# Mean = 0, Standard Deviation = 1 for each feature

# Dimensionality Reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)
# Reduced from 84 → 40 principal components
```

**Rationale**: SVMs are sensitive to feature scales and benefit from dimensionality reduction in high-dimensional sparse feature spaces.

#### For CNN Model
```python
# One-Hot Encoding per nucleotide position
sequence = "ACGT..."  # 200bp window
encoding = np.zeros((4, 200), dtype=np.float32)
mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

for position, base in enumerate(sequence):
    channel = mapping[base]
    encoding[channel, position] = 1.0

# Output shape: (4 channels × 200 positions)
```

**Rationale**: CNNs learn directly from raw sequences as 1D images, bypassing manual feature engineering.

---

### 5. Train-Test Split

#### Split Configuration
```python
train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    stratify=y,         # Maintain class balance
    random_state=42     # Reproducibility
)
```

#### Dataset Sizes
- **Training Set**: 2,035 samples
  - Benign: ~1,158 samples
  - Pathogenic: ~877 samples
- **Test Set**: 509 samples
  - Benign: 290 samples
  - Pathogenic: 219 samples

#### Stratification Importance
- Preserves 56.9% Benign / 43.1% Pathogenic ratio in both splits
- Prevents biased evaluation on imbalanced data
- Critical for medical diagnostic applications where false negatives (missing pathogenic variants) have clinical consequences

---

## Data Quality Metrics

### Completeness
- **Missing Values**: 0 (all windows have complete sequence data)
- **Reference Match Rate**: 100% (only ref_match=True retained)
- **Chromosome Coverage**: Chromosome 7 only (CFTR gene location)

### Data Statistics
```json
{
  "variants_total": 2544,
  "variants_by_label": {
    "benign": 1448,
    "pathogenic": 1096
  },
  "windows_n": 2544,
  "window_len": {
    "min": 200,
    "max": 201,
    "median": 201.0,
    "mean": 200.998
  },
  "ref_match_rate": 1.0
}
```

### Feature Matrix Dimensions
- **Rows (Samples)**: 2,544 variants
- **Columns (Features)**: 84 engineered features + 1 label
- **Sparsity**: ~25% (k-mer features sparse due to 64 possible 3-mers)
- **Data Type**: Mixed (continuous, binary, categorical)

---

## Preprocessing Validation

### Cross-Reference Checks
1. ✅ VCF reference allele matches hg38 FASTA at variant position
2. ✅ Window sequences contain expected reference allele at center
3. ✅ All features have finite values (no NaN, Inf)
4. ✅ Label distribution preserved across train/test splits
5. ✅ Feature value ranges are biologically plausible (e.g., GC content ∈ [0,1])

### Reproducibility Controls
- **Random Seed**: 42 (all splits, models, cross-validation)
- **Data Versioning**: ClinVar snapshot date recorded
- **Reference Genome**: hg38 assembly version specified
- **Code Version Control**: Git repository tracking all preprocessing scripts

---

## Clinical Relevance of Features

### Why These Features Matter for Pathogenicity Prediction

1. **Sequence Context (k-mers)**: 
   - Pathogenic variants often cluster in functional domains
   - Certain sequence motifs correlate with structural importance
   - CpG dinucleotides have elevated mutation rates

2. **GC Content**:
   - Exons tend to be GC-rich
   - GC content affects DNA stability and gene expression
   - Deviations may indicate regulatory disruption

3. **Transition/Transversion Ratio**:
   - Different mutational mechanisms leave distinct signatures
   - Pathogenic variants may show altered ratios
   - Helps distinguish true mutations from sequencing errors

4. **Sequence Symmetry**:
   - Secondary structures (hairpins, loops) affect protein binding
   - Palindromic sequences often have regulatory functions
   - Disruption correlates with functional impact

This preprocessing pipeline transforms raw genomic data into a feature-rich dataset suitable for machine learning models to learn patterns distinguishing benign from pathogenic CFTR variants.
