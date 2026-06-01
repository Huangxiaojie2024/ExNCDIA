# ExNCDIA

**Explainable prediction and mechanistic insights into non-chemotherapy drug-induced agranulocytosis (NCDIA) through ensemble machine learning.**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://exncdia-predictor.streamlit.app/)

ExNCDIA is an explainable, imbalance-aware ensemble machine-learning framework for predicting NCDIA toxicity directly from chemical structure and for interpreting the molecular features that drive the risk. This repository contains the trained models, the reproducibility notebooks, the curated datasets, and the source code of the freely accessible web predictor.

**Live predictor:** https://exncdia-predictor.streamlit.app/

![Figure 1. Overview of the ExNCDIA framework for explainable prediction of NCDIA.](ExNCDIA%20Figure.jpg)

*Figure 1. Overview of the ExNCDIA framework for explainable prediction of NCDIA.*

---

## Overview

Non-chemotherapy drug-induced agranulocytosis is a rare but potentially fatal adverse reaction whose idiosyncratic, multifactorial nature makes prospective risk assessment difficult. ExNCDIA was built on a curated dataset of 906 drugs (371 NCDIA-toxic, 535 non-toxic) assembled from literature-derived compounds and SIDER agranulocytosis annotations. Three open-source molecular representations (RDKit and Mordred descriptors, and MACCS fingerprints) were paired with five feature-selection strategies, and five imbalance-aware ensemble classifiers were trained, tuned by Matthews correlation coefficient (MCC), and validated by 10-fold cross-validation and a held-out test set. Model behavior was characterized by applicability-domain analysis and by global and local interpretation with SHapley Additive exPlanations (SHAP).

Two balanced random forest models performed best and are deployed in the web predictor: the descriptor-based **RDKit_GA_BRF** and the fingerprint-based **MACCS_FP_BRF**. SHAP analyses of these two non-overlapping representations converged on the same chemically coherent risk determinants — sulfur-containing pharmacophores, multiple aromatic rings, and nitrogen-bearing/arylamine cores — consistent with documented cytochrome P450- and myeloperoxidase-mediated bioactivation routes of neutrophil injury.

## Key results

| Representation | Evaluation | AUC | ACC | SEN | SPE | MCC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **RDKit (GA, 85)** | 10-fold CV | 0.834 | 0.759 | 0.749 | 0.767 | 0.515 |
|                    | Internal test | 0.850 | 0.797 | 0.813 | 0.785 | 0.591 |
| **MACCS (FP, 131)** | 10-fold CV | 0.821 | 0.754 | 0.752 | 0.756 | 0.503 |
|                     | Internal test | 0.847 | 0.764 | 0.747 | 0.776 | 0.518 |


Both models show well-balanced sensitivity and specificity and complete applicability-domain coverage of the test set.

## Repository structure

```
ExNCDIA/
├── app.py                       # Streamlit web app (entry point)
├── predictor.py                 # prediction logic: load model, predict class, applicability domain
├── featurizers.py               # SMILES -> molecular features (RDKit descriptors / MACCS keys)
├── explainer.py                 # SHAP explanation and waterfall plot
├── config.toml                  # Streamlit app configuration
├── packages.txt                 # system packages for Streamlit Community Cloud
├── requirements.txt             # Python dependencies for the web app
├── NCDIA_RDKit_BRF_model.pkl    # trained RDKit_GA_BRF model
├── NCDIA_MACCS_BRF_model.pkl    # trained MACCS_FP_BRF model
├── NCDIA Figure 1.jpg           # framework overview (Figure 1)
├── README.md
├── Code/                        # reproducibility notebooks
│   ├── ExNCDIA_RDKit_Prediction.ipynb
│   └── ExNCDIA_MACCS_Prediction.ipynb
└── Data Set/                    # curated training / test data
    ├── ExNCDIA_trainingset_RDKit_descriptors.csv
    ├── ExNCDIA_testset_RDKit_descriptors.csv
    ├── ExNCDIA_trainingset_MACCS_keys.csv
    ├── ExNCDIA_testset_MACCS_keys.csv
    ├── X_train_GA_RDKit.csv     # GA-selected RDKit subset (training)
    └── X_test_GA_RDKit.csv      # GA-selected RDKit subset (test)
```

## Using the online predictor

Open https://exncdia-predictor.streamlit.app/ in any browser and enter one or more compounds as SMILES strings. The app computes the required molecular features on the fly and, for each query, returns:

- NCDIA risk predictions from **both** the descriptor-based (RDKit_GA_BRF) and fingerprint-based (MACCS_FP_BRF) models, together with the estimated probability of being NCDIA-positive;
- an applicability-domain assessment indicating whether the compound falls within the chemical space spanned by the training set (and therefore whether the prediction can be regarded as reliable); and
- a SHAP waterfall plot that ranks the molecular features driving the prediction for that compound.

The interface also supports **batch prediction** of multiple compounds and lets the computed descriptors and fingerprints be exported for further analysis.

## Reproducing the models

The notebooks in `Code/` reproduce the two best models end to end (preprocessing → feature selection → hyperparameter tuning → evaluation → SHAP).

**1. Environment.** The models were developed with **Python 3.9.19**; the exact package versions are pinned in `requirements.txt` (key versions: scikit-learn 1.5.1, imbalanced-learn 0.12.3, xgboost 1.6.1, lightgbm 3.3.5, hyperopt 0.2.7, deap 1.4.1, shap 0.46.0, RDKit 2022.9.5, Mordred 1.2.0, NumPy 1.26.4, pandas 2.2.3, SciPy 1.11.2). Install them with `pip install -r requirements.txt`. Using these versions is recommended for exact reproduction (the Easy Ensemble model relies on an AdaBoost option removed in scikit-learn ≥ 1.6).

**2. Data.** The CSV files in `Data Set/` follow the layout: column 0 = PubChem CID, column 1 = Label (1 = NCDIA-toxic, 0 = non-toxic), column 2 = Data Set, column 3 = Standardized_SMILES, columns 4+ = features. Place the relevant CSVs in the notebook's working directory (or adjust the paths at the top of each notebook).

**3. Run the notebooks.**

- `ExNCDIA_RDKit_Prediction.ipynb` — Section 3 tunes the base Balanced Random Forest on the preprocessed descriptors; Section 4 documents the five feature-selection strategies and the per-subset comparison that selected the GA subset. The genetic algorithm is slow and can be skipped — the precomputed GA subset (`X_train_GA_RDKit.csv`, `X_test_GA_RDKit.csv`) is loaded instead. Section 5 tunes and evaluates the five classifiers; the final cells produce the ROC comparison and SHAP plots.
- `ExNCDIA_MACCS_Prediction.ipynb` — same structure; for the MACCS keys, feature preprocessing only (FP) was the best subset, so the base model is also the final MACCS_FP_BRF model.

Each classifier shows its Hyperopt tuning routine (as a reference template) followed by the optimal parameter combination; all reported metrics are the mean across the 10 cross-validation folds and the held-out test set.

## Running the web app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `NCDIA_RDKit_BRF_model.pkl` / `NCDIA_MACCS_BRF_model.pkl` and exposes the same SMILES-based interface as the hosted version. `packages.txt` lists the system libraries needed when deploying on Streamlit Community Cloud.

## Dataset

The curated collection comprises **906 drugs** with confirmed NCDIA outcomes (371 toxic / 535 non-toxic), partitioned by an 8:2 stratified split (`random_state = 3`) into a **training set of 724** (296 toxic / 428 non-toxic) and a **held-out test set of 182** (75 toxic / 107 non-toxic).

## Method summary

- **Representations:** RDKit descriptors (208), Mordred descriptors (1613), MACCS keys (166), computed independently.
- **Feature preprocessing (FP):** removal of missing/constant features, standardization of continuous features, variance thresholding, and Pearson-correlation pruning (|r| > 0.9).
- **Feature selection (5 strategies):** ANOVA F-test (p < 0.05), mutual information (MI > 0), RFECV, genetic algorithm (GA), and embedded tree-based selection (ETB), all using a Balanced Random Forest base estimator evaluated by 10-fold CV.
- **Classifiers (5, imbalance-aware):** Balanced Random Forest (BRF), Easy Ensemble (EEC), and Balanced Bagging wrapping GBDT, LightGBM, or XGBoost.
- **Tuning & validation:** Hyperopt (TPE) optimization of MCC; 10-fold cross-validation plus an independent test set.
- **Interpretation & deployment:** applicability-domain analysis, global and local SHAP interpretation, and release of the best models as a Streamlit web predictor.

## Citation

If you use this code, the models, or the dataset, please cite:

> Huang X, Jiang S. *ExNCDIA: Explainable prediction and mechanistic insights into non-chemotherapy drug-induced agranulocytosis through ensemble machine learning approaches.* (Manuscript; citation details to be updated upon publication.)

## Funding

This work was supported by the Medical Science and Technology Research Foundation of Guangdong Province (Grant No. A2024072).

## Contact

Xiaojie Huang — Department of Pharmacy, Jieyang People's Hospital, Jieyang 522000, China — huangxj46@alumni.sysu.edu.cn

## License

Released for academic and non-commercial research use. Please contact the corresponding author regarding other uses.
