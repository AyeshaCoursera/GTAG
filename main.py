"""Main"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules
from data import GTExLoader, TCGALoader, ExpressionPreprocessor
from analysis import TissueSpecificAnalyzer, CoexpressionAnalyzer, DatabaseValidator
from ml import TabTransformer, XGBoostWrapper, LASSOWrapper, SHAPExplainer, VAEGenerator
from visualization import StaticVisualizer, InteractiveVisualizer


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run the complete transcriptomic analysis pipeline."""
    
    # Load configuration
    config = load_config()
    np.random.seed(config['data']['random_seed'])
    
    # Create output directory
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(exist_ok=True)
    
    
    print("TRANSCRIPTOMIC VISUAL PIPELINE")
    
    # 1. DATA LOADING AND PREPROCESSING
    
    print("\n[1/6] Loading and preprocessing data...")
    
    # Load GTEx data
    gtex_loader = GTExLoader(config['data']['gtex_s3_path'])
    adata = gtex_loader.load()
    
    # Get HVG indices
    hvg_indices, hvg_names = gtex_loader.get_hvg_indices(config['data']['n_hvg_genes'])
    
    # Extract expression matrix
    n_cells = config['data']['n_cells_sample']
    cell_indices = np.random.choice(adata.shape[0], min(n_cells, adata.shape[0]), replace=False)
    expr_subset = adata[cell_indices, hvg_indices].layers['normalized']
    from scipy import sparse
    if sparse.issparse(expr_subset):
        expr_matrix = expr_subset.toarray()
    else:
        expr_matrix = expr_subset
    tissue_labels = adata.obs['tissue'].iloc[cell_indices].values
    
    print(f"Expression matrix: {expr_matrix.shape[0]} cells, {expr_matrix.shape[1]} genes")
    print(f"Unique tissues: {len(np.unique(tissue_labels))}")
    
    # Preprocess
    preprocessor = ExpressionPreprocessor(random_state=config['data']['random_seed'])
    expr_scaled = preprocessor.preprocess_gtex(expr_matrix, n_cells=n_cells)
    
    # Get donor-level data for ML
    X_donor, y_donor = preprocessor.get_donor_level_data(adata, hvg_indices)
    print(f"Donor-level data: {X_donor.shape[0]} donors, {X_donor.shape[1]} genes")
    
    # Load TCGA data
    tcga_loader = TCGALoader(config['data']['tcga_path'])
    tcga_data = tcga_loader.load()
    tcga_labels = tcga_loader.get_labels()
    
    # Preprocess TCGA
    X_tcga, y_tcga, tcga_features = preprocessor.preprocess_tcga(
        tcga_data.T, tcga_labels, n_features=2000
    )
    print(f"TCGA data: {X_tcga.shape[0]} samples, {X_tcga.shape[1]} features")
    
    
    # 2. TISSUE-SPECIFIC EXPRESSION ANALYSIS
    
    print("\n[2/6] Analyzing tissue-specific expression patterns...")
    
    tissue_analyzer = TissueSpecificAnalyzer(expr_matrix, hvg_names, tissue_labels)
    tissue_results = tissue_analyzer.analyze_tissue_specificity(
        top_n=config['analysis']['top_genes_per_tissue']
    )
    
    print(f"Analyzed {len(tissue_results)} tissues")
    for tissue, data in list(tissue_results.items())[:3]:
        print(f"  {tissue[:30]}: {data['top_genes'][:3]}")
    
    # Generate gene descriptions
    gene_descriptions = tissue_analyzer.generate_gene_descriptions(tissue_results)
    
    
    # 3. CO-EXPRESSION NETWORK ANALYSIS
    
    print("\n[3/6] Building co-expression networks...")
    
    coexpr_analyzer = CoexpressionAnalyzer(expr_matrix, hvg_names, tissue_labels)
    tissue_networks = coexpr_analyzer.find_coexpression_networks(
        corr_threshold=config['analysis']['correlation_threshold']
    )
    
    # Validate co-expression pairs statistically
    validated_pairs = {}
    for tissue in config['analysis'].get('tissues_to_validate', 
                                          ['anterior wall of left ventricle', 'esophagus muscularis mucosa']):
        pairs = coexpr_analyzer.calculate_statistical_pairs(
            tissue, corr_threshold=0.6
        )
        validated_pairs[tissue] = pairs
        fdr_pairs = [p for p in pairs if p.get('significant_fdr', False)]
        print(f"  {tissue[:30]}: {len(fdr_pairs)} FDR-significant pairs")
    
    
    # 4. MACHINE LEARNING FOR SEX PREDICTION
    
    print("\n[4/6] Training models for sex prediction...")
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_donor, y_donor, test_size=0.2, random_state=42, stratify=y_donor
    )
    
    # Scale data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # LASSO
    lasso = LASSOWrapper()
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict_proba(X_test_scaled)
    from sklearn.metrics import roc_auc_score
    auc_lasso = roc_auc_score(y_test, y_pred_lasso)
    print(f"  LASSO AUC: {auc_lasso:.4f}")
    
    # XGBoost
    xgb = XGBoostWrapper()
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict_proba(X_test_scaled)
    auc_xgb = roc_auc_score(y_test, y_pred_xgb)
    print(f"  XGBoost AUC: {auc_xgb:.4f}")
    
    # SHAP explainability
    explainer = SHAPExplainer(xgb.model, hvg_names)
    shap_values = explainer.explain(X_test_scaled)
    top_features = explainer.get_top_features(10)
    print("\n  Top 10 predictive genes:")
    print(top_features.to_string(index=False))
    
    
    # 5. TUMOR/NORMAL CLASSIFICATION (TCGA)
    
    print("\n[5/6] Training models for tumor/normal classification...")
    
    # Split TCGA data
    X_train_tcga, X_test_tcga, y_train_tcga, y_test_tcga = train_test_split(
        X_tcga, y_tcga, test_size=0.2, random_state=42, stratify=y_tcga
    )
    
    # Scale
    scaler_tcga = StandardScaler()
    X_train_tcga_scaled = scaler_tcga.fit_transform(X_train_tcga)
    X_test_tcga_scaled = scaler_tcga.transform(X_test_tcga)
    
    # LASSO
    lasso_tcga = LASSOWrapper()
    lasso_tcga.fit(X_train_tcga_scaled, y_train_tcga)
    y_pred_lasso_tcga = lasso_tcga.predict_proba(X_test_tcga_scaled)
    auc_lasso_tcga = roc_auc_score(y_test_tcga, y_pred_lasso_tcga)
    print(f"  LASSO AUC: {auc_lasso_tcga:.4f}")
    
    # XGBoost
    xgb_tcga = XGBoostWrapper()
    xgb_tcga.fit(X_train_tcga_scaled, y_train_tcga)
    y_pred_xgb_tcga = xgb_tcga.predict_proba(X_test_tcga_scaled)
    auc_xgb_tcga = roc_auc_score(y_test_tcga, y_pred_xgb_tcga)
    print(f"  XGBoost AUC: {auc_xgb_tcga:.4f}")
    
    
    # 6. VAE FOR SYNTHETIC DATA GENERATION
    
    print("\n[6/6] Training VAE for synthetic data generation...")
    
    vae = VAEGenerator(
        input_dim=expr_scaled.shape[1],
        latent_dim=config['vae']['latent_dim'],
        beta=config['vae']['beta']
    )
    vae.fit(expr_scaled, batch_size=config['vae']['batch_size'],
            epochs=config['vae']['num_epochs'])
    
    # Generate synthetic samples
    synthetic_expr = vae.generate(500)
    print(f"  Generated {synthetic_expr.shape[0]} synthetic cell profiles")
    
    
    # VISUALIZATION
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    
    
    static_viz = StaticVisualizer()
    interactive_viz = InteractiveVisualizer(output_dir=str(output_dir))
    
    # Static plots
    static_viz.plot_tissue_heatmap(tissue_results, output_path=str(output_dir / "tissue_heatmap.png"))
    static_viz.plot_network_complexity(tissue_networks, output_path=str(output_dir / "network_complexity.png"))
    static_viz.plot_pca(expr_matrix, tissue_labels, output_path=str(output_dir / "pca.png"))
    
    # Interactive plots
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(expr_scaled[:1000])
    interactive_viz.pca_plot(pca_result, tissue_labels[:1000])
    
    # Save results
    results = {
        'tissue_specific_genes': tissue_results,
        'coexpression_networks': tissue_networks,
        'gene_descriptions': gene_descriptions,
        'sex_prediction': {'lasso_auc': auc_lasso, 'xgboost_auc': auc_xgb},
        'tcga_classification': {'lasso_auc': auc_lasso_tcga, 'xgboost_auc': auc_xgb_tcga},
        'top_predictive_genes': top_features.to_dict('records')
    }
    
    import json
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print("PIPELINE COMPLETE!")
    print(f"Results saved to: {output_dir}")
    


if __name__ == "__main__":
    main()
