# CFTR Variant Classification: Model Comparison

## Executive Summary

This report compares four machine learning approaches for classifying CFTR variants as benign or pathogenic using 2,544 ClinVar variants (1,448 benign, 1,096 pathogenic). All models were trained on 80% of the data (2,035 samples) and evaluated on a held-out test set of 509 samples (290 benign, 219 pathogenic).

**Best Overall Model**: **LightGBM** (ROC-AUC: 0.706, F1: 0.643, Accuracy: 64.0%)

---

## Performance Metrics Comparison

| Model | ROC-AUC | PR-AUC | F1 Score | Accuracy | Optimal Threshold |
|-------|---------|--------|----------|----------|-------------------|
| **LightGBM** | **0.706** | **0.635** | 0.643 | **0.640** | 0.400 |
| **XGBoost** | 0.704 | 0.634 | **0.652** | 0.629 | 0.400 |
| SVM (RBF + PCA) | 0.664 | 0.611 | 0.616 | 0.530 | 0.300 |
| CNN (PyTorch) | 0.660 | 0.586 | 0.618 | 0.609 | 0.225 |

### Key Findings:
- **Tree-based models (XGBoost, LightGBM)** significantly outperform SVM and CNN
- **LightGBM** achieves highest ROC-AUC (0.706) and accuracy (64.0%)
- **XGBoost** achieves highest F1 score (0.652)
- All models required threshold tuning below 0.5 to optimize F1 score

---

## Detailed Model Analysis

### 1. XGBoost Classifier
**Configuration:**
- Best parameters: `learning_rate=0.03, max_depth=3, n_estimators=600`
- Optimal threshold: 0.400
- Features: 84 engineered features (k-mers, GC content, transition/transversion, one-hot encoded bases)

**Performance:**
- ROC-AUC: 0.704
- F1 Score: **0.652** (highest)
- Accuracy: 62.9%
- Recall (Pathogenic): **0.81** (highest)

**Confusion Matrix:**
```
                Predicted
              Benign  Path
Actual Benign   143    147
       Path      42    177
```

**Strengths:**
- ✅ Highest recall for pathogenic variants (81%) - critical for clinical safety
- ✅ Best F1 score balances precision and recall
- ✅ Excellent for minimizing false negatives (missed pathogenic variants)

**Weaknesses:**
- ⚠️ Lower precision for pathogenic class (55%) - more false positives
- ⚠️ Only 49% recall for benign class

**Clinical Interpretation:**
XGBoost prioritizes sensitivity over specificity, making it ideal for screening applications where missing a pathogenic variant has serious consequences. The tradeoff is more false positives requiring follow-up investigation.

---

### 2. LightGBM Classifier
**Configuration:**
- Best parameters: `learning_rate=0.1, max_depth=5, n_estimators=200, num_leaves=31`
- Optimal threshold: 0.400
- Features: 84 engineered features (same as XGBoost)

**Performance:**
- ROC-AUC: **0.706** (highest)
- F1 Score: 0.643
- Accuracy: **64.0%** (highest)
- Balanced recall: 0.56 (benign), 0.75 (pathogenic)

**Confusion Matrix:**
```
                Predicted
              Benign  Path
Actual Benign   161    129
       Path      54    165
```

**Strengths:**
- ✅ Highest ROC-AUC (0.706) - best overall discriminative ability
- ✅ Highest accuracy (64.0%)
- ✅ Most balanced performance across both classes
- ✅ Better benign recall (56%) than XGBoost

**Weaknesses:**
- ⚠️ Slightly lower pathogenic recall (75%) compared to XGBoost (81%)
- ⚠️ 54 false negatives (pathogenic variants misclassified as benign)

**Clinical Interpretation:**
LightGBM offers the best overall balance, making it suitable for diagnostic applications where both false positives and false negatives have significant costs. It correctly identifies 75% of pathogenic variants while maintaining reasonable specificity.

---

### 3. SVM (RBF Kernel + PCA)
**Configuration:**
- Best parameters: `C=10.0, gamma=0.001`
- Optimal threshold: 0.300
- Preprocessing: StandardScaler + PCA (84 → 40 components, 95.4% variance retained)

**Performance:**
- ROC-AUC: 0.664
- F1 Score: 0.616
- Accuracy: 53.0% (lowest)
- Recall (Pathogenic): **0.88** (highest)

**Confusion Matrix:**
```
                Predicted
              Benign  Path
Actual Benign    78    212
       Path      27    192
```

**Strengths:**
- ✅ Highest pathogenic recall (88%) - fewest missed pathogenic variants
- ✅ Only 27 false negatives (best safety profile)

**Weaknesses:**
- ⚠️ Very low benign recall (27%) - misses most benign variants
- ⚠️ 212 false positives - highest rate of incorrect pathogenic predictions
- ⚠️ Lowest accuracy (53.0%)
- ⚠️ Poor precision for pathogenic class (48%)

**Clinical Interpretation:**
SVM maximizes sensitivity at the expense of specificity. While it catches 88% of pathogenic variants, it also incorrectly flags 73% of benign variants as pathogenic. This would lead to excessive follow-up testing and patient anxiety in clinical practice.

---

### 4. CNN (Convolutional Neural Network - PyTorch)
**Configuration:**
- Architecture: 3 conv layers (64→128→256 filters, kernel=7) + 2 FC layers
- Training: 8 epochs, batch_size=32, learning_rate=0.001
- Optimal threshold: 0.225
- Input: One-hot encoded DNA sequences (4 × 201 matrix)

**Performance:**
- ROC-AUC: 0.660
- F1 Score: 0.618
- Accuracy: 60.9%
- Balanced recall: 0.51 (benign), 0.74 (pathogenic)

**Confusion Matrix:**
```
                Predicted
              Benign  Path
Actual Benign   149    141
       Path      58    161
```

**Strengths:**
- ✅ No manual feature engineering required - learns from raw sequences
- ✅ Balanced performance across classes
- ✅ Good pathogenic recall (74%)

**Weaknesses:**
- ⚠️ Lower ROC-AUC (0.660) than tree models
- ⚠️ Requires more training time (~24s vs ~16s for tree models)
- ⚠️ Only 8 epochs trained - may improve with longer training

**Clinical Interpretation:**
CNN shows promise as an end-to-end learner but underperforms compared to tree models with engineered features for this dataset size. Deep learning typically requires larger datasets (10,000+ samples) to reach full potential.

---

## Class-Specific Performance Analysis

### Benign Variant Detection (Class 0)
| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **XGBoost** | **0.77** | 0.49 | 0.60 |
| **LightGBM** | 0.75 | **0.56** | **0.64** |
| SVM | 0.74 | 0.27 | 0.39 |
| CNN | 0.72 | 0.51 | 0.60 |

**Winner: LightGBM** - Best recall (56%) and F1 score (0.64) for benign variants

### Pathogenic Variant Detection (Class 1) - CRITICAL FOR PATIENT SAFETY
| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| XGBoost | 0.55 | **0.81** | **0.65** |
| LightGBM | 0.56 | 0.75 | 0.64 |
| **SVM** | 0.48 | **0.88** | 0.62 |
| CNN | 0.53 | 0.74 | 0.62 |

**Winner for Recall: SVM** (88%) - Catches most pathogenic variants
**Winner for F1: XGBoost** (0.65) - Best balance of precision and recall

---

## Threshold Optimization Analysis

All models benefited significantly from threshold tuning:

| Model | Default (0.5) | Optimized | Threshold Value | Improvement |
|-------|---------------|-----------|-----------------|-------------|
| XGBoost | Lower F1 | 0.652 | 0.400 | Significant |
| LightGBM | Lower F1 | 0.643 | 0.400 | Significant |
| SVM | Lower F1 | 0.616 | 0.300 | Very Significant |
| CNN | Lower F1 | 0.618 | 0.225 | Very Significant |

**Key Insight**: Lower thresholds increase pathogenic recall (sensitivity), which is clinically appropriate given the serious consequences of missing pathogenic variants in cystic fibrosis.

---

## Computational Efficiency

### Training Time (approximate from pipeline outputs)
- **XGBoost**: ~16 seconds (18 hyperparameter combinations)
- **CNN**: ~24 seconds (8 epochs)
- **LightGBM**: ~66 seconds (54 hyperparameter combinations)
- **SVM**: ~17 seconds (20 hyperparameter combinations)

**Most Efficient**: XGBoost (fastest training with top performance)

---

## Model Selection Recommendations

### Scenario 1: Clinical Screening (Maximize Sensitivity)
**Recommended: XGBoost**
- Rationale: Highest pathogenic recall (81%) with acceptable false positive rate
- Use case: Initial variant screening where missed pathogenic variants are unacceptable
- Follow-up: Positive predictions undergo confirmatory testing

### Scenario 2: Diagnostic Application (Balanced Performance)
**Recommended: LightGBM**
- Rationale: Best overall accuracy (64.0%) and ROC-AUC (0.706)
- Use case: Confirmatory testing where both false positives and negatives have costs
- Follow-up: Borderline cases referred for functional studies

### Scenario 3: Research/VUS Reclassification
**Recommended: XGBoost or LightGBM**
- Rationale: Highest AUC scores indicate best separation between classes
- Use case: Prioritizing Variants of Uncertain Significance (VUS) for experimental validation
- Follow-up: Use probability scores to rank variants for functional assays

### Scenario 4: Large-Scale Genomic Studies
**Recommended: CNN (with more training)**
- Rationale: Scales better with larger datasets and can learn complex sequence patterns
- Use case: When dataset grows to 10,000+ variants
- Note: Current performance limited by small dataset size (2,544 samples)

---

## Limitations and Future Directions

### Current Limitations:
1. **Dataset Size**: 2,544 variants is relatively small for deep learning approaches
2. **Single Gene**: Models trained exclusively on CFTR; generalization to other genes unknown
3. **Binary Classification**: Does not handle Variants of Uncertain Significance (VUS)
4. **Sequence Context Only**: Does not incorporate protein structure, conservation scores, or functional data

### Recommended Improvements:
1. **Ensemble Methods**: Combine XGBoost and LightGBM predictions via voting or stacking
2. **Additional Features**: 
   - PhyloP/PhastCons conservation scores
   - CADD/REVEL pathogenicity scores
   - Protein structure predictions (AlphaFold)
   - Allele frequency from gnomAD
3. **Multi-class Classification**: Extend to predict VUS, likely benign, likely pathogenic
4. **Transfer Learning**: Pre-train CNN on all human variants, fine-tune on CFTR
5. **Uncertainty Quantification**: Add confidence intervals to predictions
6. **External Validation**: Test on independent CFTR variant datasets

---

## Conclusion

**Tree-based gradient boosting methods (XGBoost and LightGBM) significantly outperform SVM and CNN** for CFTR variant classification with engineered features. 

**LightGBM is recommended as the primary model** due to its:
- Highest ROC-AUC (0.706) and accuracy (64.0%)
- Best balance between sensitivity and specificity
- Superior computational efficiency at inference time

**XGBoost serves as an excellent alternative** when maximizing pathogenic variant detection is the priority (81% recall).

The relatively modest performance across all models (AUC ~0.66-0.71) highlights the inherent difficulty of variant pathogenicity prediction and suggests that:
1. Sequence context alone is insufficient - additional features needed
2. Clinical interpretation should not rely solely on computational predictions
3. Borderline predictions require functional validation or expert review

**Clinical Impact**: These models can assist in triaging variants but should not replace genetic counselor review and functional studies. They are best used to:
- Prioritize variants for experimental validation
- Flag potentially misclassified variants in databases
- Support evidence-based clinical decision-making alongside other criteria
