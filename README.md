# Transcriptomic Analysis 

A comprehensive Python framework for analyzing single-nucleus RNA-seq data from GTEx v9 and bulk RNA-seq data from TCGA-STAD. This pipeline performs tissue-specific gene expression analysis, co-expression network inference, machine learning-based classification (gender prediction, tumor/normal), explainable AI (SHAP), and generative modeling (VAE) for synthetic data generation.

## Installation

### Requirements

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Device

- **Development**: Single NVIDIA GPU (CUDA 11+) recommended for model training
- **Inference**: CPU or GPU (minimum 16GB RAM for GTEx dataset)

## Key Algorithms and Implementation

### 1. Tissue-Specific Gene Expression Analysis

The framework identifies tissue-enriched genes using mean expression ranking across cell populations from GTEx v9 single-nucleus data.

- **Input**: Normalized expression matrix (cells × genes) with tissue annotations
- **Method**: Per‑tissue mean expression calculation and ranking
- **Output**: Top‑N genes per tissue with expression values and cell counts

### 2. Co‑expression Network Inference

Tissue‑specific gene co‑expression networks are built using Pearson correlation, followed by FDR correction (Benjamini‑Hochberg) to identify statistically significant gene pairs.

- **Correlation threshold**: |r| > 0.65 (user‑adjustable)
- **Statistical validation**: FDR‑adjusted p‑value < 0.05
- **Output**: List of significant co‑expression pairs per tissue with correlation coefficients and adjusted p‑values

### 3. Machine Learning for Gender Prediction (GTEx)

A **TabTransformer** architecture is used for donor‑level gender classification based on gene expression profiles. XGBoost and LASSO serve as baselines.

- **Input**: Donor‑aggregated expression of highly variable genes (HVGs)
- **Model**: TabTransformer (self‑attention over features), XGBoost, LASSO
- **Explainability**: SHAP values highlight genes driving gender prediction

### 4. Tumor vs. Normal Classification (TCGA‑STAD)

DNN, XGBoost, and LASSO models classify stomach adenocarcinoma samples using RNA‑seq counts.

- **Preprocessing**: Variance filtering, ANOVA feature selection (top 2000 genes), standard scaling
- **Models**: DNN (batch norm + dropout), XGBoost (scale_pos_weight for imbalance), LASSO (L1 penalty)
- **Evaluation**: ROC‑AUC, accuracy, F1 score

### 5. Variational Autoencoder (VAE) for Synthetic Data Generation

A **beta‑VAE** learns a latent representation of GTEx expression profiles and generates realistic synthetic single‑cell data. Transfer learning enables augmentation of rare tissues (e.g., skin of leg).

- **Architecture**: Encoder + decoder with batch normalization, reparameterization trick
- **Loss**: MSE reconstruction + beta × KL divergence
- **Quality assessment**: Wasserstein distance, t‑SNE visualization

## Data Preparation

### GTEx v9 Single‑Cell Atlas

The dataset is loaded directly from S3 using anonymous access:

```python
from data import GTExLoader
loader = GTExLoader("s3://openproblems-data/resources/datasets/cellxgene_census/gtex_v9/log_cp10k/dataset.h5ad")
adata = loader.load()
```

### TCGA‑STAD

Place the file `TCGA-STAD.star_counts.tsv` (downloaded from GDC) in the project root or `data/` directory.

### File Structure

```
transcriptomic-framework/
├── __init__.py
├── config.yaml                 # All parameters (paths, thresholds, random seed)
├── data/
│   ├── __init__.py
│   ├── loader.py               # GTEx and TCGA data loaders
│   └── preprocessor.py         # Normalization, HVG selection, QC
├── analysis/
│   ├── __init__.py
│   ├── tissue_specific.py      # LFC + FDR for gene discovery
│   ├── coexpression.py         # Correlation + FDR for networks
│   └── validation.py           # Database API queries (Ensembl, etc.)
├── ml/
│   ├── __init__.py
│   ├── classifier.py           # TabTransformer, XGBoost, LASSO wrappers
│   ├── explainer.py            # SHAP analysis module
│   └── generator.py            # VAE for synthetic data
├── visualization/
│   ├── __init__.py
│   ├── plots.py                # All matplotlib/seaborn visualizations
│   └── interactive.py          # Plotly/Dash interactive dashboards
├── main.py                     # Orchestrates the full pipeline
├── requirements.txt
└── README.md
```

## Usage

### Quick Start (Full Pipeline)

Run the complete analysis with default settings:
```bash
python main.py
```

You can modify parameters in `config.yaml` before running.

### Individual Modules

#### Tissue‑Specific Analysis & Co‑expression
```python
from analysis import TissueSpecificAnalyzer, CoexpressionAnalyzer

tissue_analyzer = TissueSpecificAnalyzer(expr_matrix, gene_names, tissue_labels)
tissue_results = tissue_analyzer.analyze_tissue_specificity(top_n=15)

coexpr_analyzer = CoexpressionAnalyzer(expr_matrix, gene_names, tissue_labels)
validated_pairs = coexpr_analyzer.calculate_statistical_pairs("anterior wall of left ventricle")
```

#### Train Gender‑Prediction Models
```python
from ml import XGBoostWrapper, LASSOWrapper
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_donor, y_donor, test_size=0.2)
xgb = XGBoostWrapper().fit(X_train, y_train)
y_pred = xgb.predict_proba(X_test)
```

#### Generate Synthetic Data with VAE
```python
from ml import VAEGenerator

vae = VAEGenerator(input_dim=expr_scaled.shape[1])
vae.fit(expr_scaled, epochs=50)
synthetic_expr = vae.generate(500)
```

#### SHAP Explainability
```python
from ml import SHAPExplainer

explainer = SHAPExplainer(xgb.model, gene_names)
shap_values = explainer.explain(X_test)
top_genes = explainer.get_top_features(10)
```

## Output Files

After a successful run, the `outputs/` directory contains:
- `analysis_results.json` – Complete results (tissue genes, co‑expression, model metrics)
- `tissue_heatmap.png` – Heatmap of tissue‑specific expression
- `network_complexity.png` – Bar plot of co‑expression pairs per tissue
- `pca.png` – PCA visualization of tissue separation
- `pca_plot.html` – Interactive PCA plot (Plotly)
- `heatmap.html` – Interactive expression heatmap
- `shap_plot.html` – Interactive SHAP summary
- `best_tabtransformer.pth` – Saved TabTransformer weights

## Acknowledgement

This framework uses the following public resources:
- **GTEx v9 Single‑Cell Atlas** – Provided by the OpenProblems consortium
- **TCGA‑STAD** – Data from the Genomic Data Commons (GDC)
- **PyTorch** – Deep learning framework
- **scikit‑learn, XGBoost, SHAP** – Machine learning and explainability
- **Scanpy, anndata** – Single‑cell data handling
- **Plotly** – Interactive visualizations
```
