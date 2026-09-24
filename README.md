# ExNCDIA

**ExNCDIA: An Explainable Ensemble Learning Framework for Predicting Non-Chemotherapy Drug-Induced Agranulocytosis and Interpreting Structural Associations**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://exncdia-predictor.streamlit.app/)

ExNCDIA is an explainable, imbalance-aware ensemble machine-learning framework for predicting non-chemotherapy drug-induced agranulocytosis (NCDIA) liability directly from chemical structure and for interpreting structural features associated with model predictions. This repository contains the trained final models, reproducibility notebooks, curated datasets, and source code for the freely accessible web predictor.

![Figure 1. Overview of the ExNCDIA framework for explainable prediction of NCDIA.](Figure%201.jpg)

*Figure 1. Overview of the ExNCDIA framework for explainable prediction of NCDIA.*

---

## Overview

Non-chemotherapy drug-induced agranulocytosis (NCDIA) is a rare but potentially life-threatening idiosyncratic adverse reaction whose mechanisms remain incompletely understood. ExNCDIA was developed to investigate whether clinically documented NCDIA liability can be predicted from molecular structure and to identify structural features associated with model predictions.

The study was built on a rigorously curated dataset of **906 drugs**, comprising **371 NCDIA-positive** and **535 NCDIA-negative** compounds. These are operational modeling labels. NCDIA-positive compounds had documented evidence of NCDIA from the literature-derived dataset and/or SIDER agranulocytosis annotations. NCDIA-negative compounds were drugs for which no qualifying NCDIA association was identified in the sources examined after application of the predefined exclusion and cross-checking procedures. Accordingly, the NCDIA-negative designation should **not** be interpreted as confirmed absence of clinical NCDIA risk.

Four complementary open-source molecular representations were evaluated independently:

- **RDKit molecular descriptors:** 208 features
- **Mordred descriptors:** 1613 features
- **MACCS structural keys:** 166 binary keys
- **ECFP4 fingerprints:** 2048 bits, Morgan radius = 2

Following feature preprocessing, five complementary feature-selection strategies were evaluated:

- analysis-of-variance F-test;
- mutual information (MI);
- recursive feature elimination with cross-validation (RFECV);
- genetic algorithm (GA); and
- embedded tree-based feature selection (ETB).

For each representation, the preprocessing-only feature set and the five feature-selection strategies were compared using a Balanced Random Forest classifier and 10-fold cross-validation on the training set, with Matthews correlation coefficient (MCC) as the primary evaluation metric.

The optimal representation-specific feature sets were:

| Representation | Selected strategy | Retained features | CV ROC-AUC | CV BA | CV SEN | CV SPE | CV MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| RDKit | GA | 85 | 0.834 ± 0.049 | 0.758 ± 0.047 | 0.749 ± 0.101 | 0.768 ± 0.078 | 0.515 ± 0.091 |
| Mordred | GA | 216 | 0.807 ± 0.060 | 0.738 ± 0.071 | 0.763 ± 0.084 | 0.714 ± 0.102 | 0.472 ± 0.140 |
| MACCS | FP | 131 | 0.821 ± 0.042 | 0.753 ± 0.055 | 0.752 ± 0.075 | 0.755 ± 0.085 | 0.503 ± 0.109 |
| ECFP4 | GA | 986 | 0.816 ± 0.049 | 0.743 ± 0.038 | 0.675 ± 0.077 | 0.812 ± 0.100 | 0.499 ± 0.089 |

The **RDKit_GA** and **MACCS_FP** feature sets were retained for subsequent classifier benchmarking. For each selected feature set, 15 classifiers were evaluated, comprising five imbalance-aware ensemble classifiers together with standard and class-weighted algorithm-matched baselines.

Model selection was primarily guided by cross-validated MCC, with sensitivity and the absolute sensitivity–specificity difference used as secondary criteria when MCC values were similar. On this basis, two Balanced Random Forest models were retained as the final models:

- **RDKit_GA_BRF** — descriptor-based model using 85 GA-selected RDKit descriptors
- **MACCS_FP_BRF** — fingerprint-based model using 131 preprocessed MACCS keys

SHapley Additive exPlanations (SHAP) were used for global and compound-level interpretation. Across the two non-overlapping molecular representations, the analyses identified reproducible structural associations involving sulfur-containing pharmacophores, multiple aromatic rings, and nitrogen-bearing/arylamine motifs. These SHAP-derived patterns represent associations with model predictions rather than evidence of causal toxicity mechanisms; their mechanistic interpretation is therefore considered in the context of previously reported bioactivation chemistry.

---

## Key results

### Final model performance

| Model | Evaluation | ROC-AUC | BA | SEN | SPE | MCC |
|---|---|---:|---:|---:|---:|---:|
| **RDKit_GA_BRF** | 10-fold CV | 0.834 ± 0.049 | 0.758 ± 0.047 | 0.749 ± 0.101 | 0.768 ± 0.078 | 0.515 ± 0.091 |
|  | Held-out test set | 0.850 | 0.799 | 0.813 | 0.785 | 0.591 |
| **MACCS_FP_BRF** | 10-fold CV | 0.821 ± 0.042 | 0.753 ± 0.055 | 0.752 ± 0.075 | 0.755 ± 0.085 | 0.503 ± 0.109 |
|  | Held-out test set | 0.847 | 0.761 | 0.747 | 0.776 | 0.518 |

For the held-out test set, 95% confidence intervals were obtained from 5000 stratified bootstrap resamples:

- **RDKit_GA_BRF:** ROC-AUC 0.850 (95% CI 0.791–0.903); BA 0.799 (0.739–0.857); SEN 0.813 (0.720–0.893); SPE 0.785 (0.710–0.860); MCC 0.591 (0.472–0.706).
- **MACCS_FP_BRF:** ROC-AUC 0.847 (95% CI 0.788–0.899); BA 0.761 (0.699–0.824); SEN 0.747 (0.653–0.840); SPE 0.776 (0.692–0.850); MCC 0.518 (0.396–0.643).

Both final models showed balanced class-wise performance on the held-out test set.

All **182 held-out test compounds** fell within the model-specific applicability domains under the applicability-domain procedure used in the study. This indicates that the held-out compounds were structurally supported by the corresponding training sets under the predefined domain criteria. It should not be interpreted as external validation, because the held-out test set was sampled from the same curated compound collection as the training set.

---

## Structural interpretation

SHAP analysis was performed using the final classifiers fitted on the complete training set and subsequently applied to the held-out test set.

The leading SHAP-ranked features included:

### RDKit_GA_BRF

- **BCUT2D_MRHI**
- **qed**
- **NumAromaticRings**
- **SlogP_VSA10**
- **VSA_EState7**

### MACCS_FP_BRF

- **MACCS_132**
- **MACCS_125**
- **MACCS_81**
- **MACCS_79**
- **MACCS_133**

Together, these features highlighted several recurring structural patterns associated with model predictions, including sulfur-containing pharmacophores, aromatic-ring content, and nitrogen-bearing or arylamine-related motifs.

Additional robustness analyses examined the stability of the leading SHAP features across alternative feature-selection settings and across ten cross-validation folds of the training set.

The resulting structural patterns should be interpreted as **model-derived associations**. Proposed links to metabolic bioactivation, reactive-metabolite formation, and immune-mediated pathways are literature-supported mechanistic hypotheses and should not be interpreted as causal mechanisms established by SHAP itself.

---

## Repository structure

```text
ExNCDIA/
├── app.py                       # Streamlit web app (entry point)
├── predictor.py                 # model loading, prediction, and applicability-domain evaluation
├── featurizers.py               # SMILES -> molecular features
├── explainer.py                 # SHAP explanations and waterfall plots
├── config.toml                  # Streamlit application configuration
├── packages.txt                 # system packages for Streamlit Community Cloud
├── requirements.txt             # Python dependencies
├── NCDIA_RDKit_BRF_model.pkl    # trained RDKit_GA_BRF model
├── NCDIA_MACCS_BRF_model.pkl    # trained MACCS_FP_BRF model
├── ExNCDIA Figure.jpg           # framework overview
├── README.md
├── Code/
│   ├── ExNCDIA_RDKit_Prediction.ipynb
│   └── ExNCDIA_MACCS_Prediction.ipynb
└── Data Set/
    ├── ExNCDIA_trainingset_RDKit_descriptors.csv
    ├── ExNCDIA_testset_RDKit_descriptors.csv
    ├── ExNCDIA_trainingset_MACCS_keys.csv
    ├── ExNCDIA_testset_MACCS_keys.csv
    ├── X_train_GA_RDKit.csv
    └── X_test_GA_RDKit.csv
```

The deployment assets and reproducibility notebooks in this repository focus on the two final models selected for the ExNCDIA web predictor. Comparative analyses involving RDKit, Mordred, MACCS, and ECFP4 representations are reported in the manuscript and Supporting Information.

---

# Using the online predictor

The ExNCDIA web application is freely accessible at:

**https://exncdia-predictor.streamlit.app/**

No local installation or programming environment is required.

## Single-compound prediction

1. Open the ExNCDIA web application.
2. Enter a valid **SMILES** string for the compound of interest.
3. Submit the query.
4. The application automatically computes the molecular features required by the two deployed models.
5. Results are returned separately for:
   - **RDKit_GA_BRF**
   - **MACCS_FP_BRF**

For each model, the application reports:

- the predicted NCDIA class;
- the corresponding **NCDIA-positive model score**;
- the applicability-domain assessment; and
- a compound-level SHAP waterfall plot.

---

## Interpretation of the predicted class

The model returns one of two operational classes:

- **NCDIA-positive**
- **NCDIA-negative**

These predictions correspond to the operational class definitions used during model development.

A predicted NCDIA-negative result should therefore **not** be interpreted as proof that a compound cannot cause agranulocytosis.

---

## Interpretation of the model score

The **NCDIA-positive model score** is the classifier output associated with prediction of the NCDIA-positive class.

It should **not** be interpreted as a calibrated absolute probability that a patient exposed to the compound will develop NCDIA.

The models are based on two-dimensional molecular structure and do not directly incorporate patient-specific determinants such as:

- dose;
- systemic exposure;
- pharmacokinetics;
- treatment duration;
- concomitant medications;
- immune status;
- genetic susceptibility; or
- other host-specific risk factors.

Accordingly, ExNCDIA is intended for structure-based research, compound prioritization, and hypothesis generation rather than direct patient-level risk estimation.

---

## Applicability-domain result

For each query compound, ExNCDIA evaluates whether the molecule lies within the structural domain represented by the corresponding training set.

An **in-domain** result indicates that the compound falls within the predefined structural applicability domain of the model.

An **out-of-domain** result indicates that the compound lies beyond the structural space represented by the training data and that the prediction therefore involves extrapolation.

Applicability-domain membership provides information about structural support relative to the training set. It does not provide a calibrated probability of prediction correctness and should not be interpreted as a direct measure of clinical certainty.

---

## SHAP waterfall plot

Each individual prediction can be accompanied by a SHAP waterfall plot.

The plot begins from the model's baseline output and illustrates how individual molecular features shift the model output toward the NCDIA-positive or NCDIA-negative class.

In general:

- positive SHAP contributions shift the model output toward **NCDIA-positive**;
- negative SHAP contributions shift the model output toward **NCDIA-negative**.

For the RDKit model, the displayed features are continuous molecular descriptors.

For the MACCS model, the displayed features are binary structural keys corresponding to predefined molecular patterns.

SHAP explains how the fitted model uses the available molecular features. It does not establish that a highlighted molecular feature is itself a causal determinant of agranulocytosis.

---

# Batch prediction

The web interface supports batch evaluation of multiple compounds.

When using batch mode:

1. prepare the compounds in the input format required by the application;
2. provide a valid SMILES representation for each compound;
3. submit the batch;
4. inspect predictions from both deployed models; and
5. export the generated results and/or molecular features when required.

Batch prediction is intended to facilitate rapid structure-based screening and prioritization of compound collections.

Before interpreting batch results, compounds should be checked for valid and chemically meaningful SMILES strings. Predictions for structures falling outside the applicability domain should be interpreted as extrapolations beyond the structural space represented by the corresponding training data.

---

# Exporting molecular features

The application supports export of molecular descriptors and fingerprints calculated for query compounds for further analysis.

Depending on the selected model, these features correspond to:

- RDKit molecular descriptors used by the descriptor-based pipeline; or
- MACCS structural keys used by the fingerprint-based pipeline.

The exported features may be used for independent inspection, downstream cheminformatics analysis, or reproduction of model inputs.

---

# Local installation

## 1. Clone the repository

```bash
git clone https://github.com/Huangxiaojie2024/ExNCDIA.git
cd ExNCDIA
```

## 2. Recommended Python environment

The models were developed with **Python 3.9.19**.

Using the same Python and package versions is recommended for reproducibility.

Key package versions include:

```text
scikit-learn       1.5.1
imbalanced-learn   0.12.3
xgboost            1.6.1
lightgbm           3.3.5
hyperopt           0.2.7
deap               1.4.1
shap               0.46.0
RDKit              2022.9.5
Mordred            1.2.0
NumPy              1.26.4
pandas             2.2.3
SciPy              1.11.2
```

The exact dependencies required for the web application are listed in `requirements.txt`.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

`packages.txt` contains the system-level libraries used for deployment on Streamlit Community Cloud.

## 4. Run the web application locally

```bash
streamlit run app.py
```

After Streamlit starts, open the local address displayed in the terminal.

The application loads the two serialized final models:

```text
NCDIA_RDKit_BRF_model.pkl
NCDIA_MACCS_BRF_model.pkl
```

and exposes the same SMILES-based prediction and interpretation workflow as the hosted web application.

---

# Reproducing the models

The notebooks in `Code/` document the development and evaluation workflows for the two final deployed models.

## 1. Data layout

The model-development CSV files follow the general structure:

```text
column 0 = PubChem CID
column 1 = Label
column 2 = Data Set
column 3 = Standardized_SMILES
columns 4+ = molecular features
```

The operational class labels are:

```text
1 = NCDIA-positive
0 = NCDIA-negative
```

The label `0` denotes the operational NCDIA-negative class used for model development and should not be interpreted as confirmed clinical safety.

The final curated collection contains:

```text
Total dataset: 906 compounds
├── NCDIA-positive: 371
└── NCDIA-negative: 535

Training set: 724 compounds
├── NCDIA-positive: 296
└── NCDIA-negative: 428

Held-out test set: 182 compounds
├── NCDIA-positive: 75
└── NCDIA-negative: 107
```

---

## 2. RDKit model

Notebook:

```text
Code/ExNCDIA_RDKit_Prediction.ipynb
```

The notebook documents the descriptor-based modeling workflow, including:

- feature preprocessing;
- comparison of feature-selection strategies;
- genetic-algorithm feature selection;
- classifier benchmarking;
- Hyperopt-based hyperparameter optimization;
- cross-validation;
- held-out test-set evaluation; and
- SHAP interpretation.

The genetic algorithm selected **85 RDKit descriptors**, producing the `RDKit_GA` feature set used by the final **RDKit_GA_BRF** model.

Because genetic-algorithm optimization is computationally intensive, the repository provides precomputed GA-selected feature matrices:

```text
Data Set/X_train_GA_RDKit.csv
Data Set/X_test_GA_RDKit.csv
```

These files can be loaded directly when re-running the downstream modeling pipeline without repeating the complete GA search.

---

## 3. MACCS model

Notebook:

```text
Code/ExNCDIA_MACCS_Prediction.ipynb
```

For MACCS keys, preprocessing without an additional feature-selection step produced the highest mean cross-validated MCC among the evaluated MACCS feature-selection settings.

The resulting **131-key MACCS_FP feature set** was therefore retained for classifier benchmarking and development of the final **MACCS_FP_BRF** model.

The corresponding input files are:

```text
Data Set/ExNCDIA_trainingset_MACCS_keys.csv
Data Set/ExNCDIA_testset_MACCS_keys.csv
```

---

## 4. RDKit descriptor files

The RDKit descriptor datasets are:

```text
Data Set/ExNCDIA_trainingset_RDKit_descriptors.csv
Data Set/ExNCDIA_testset_RDKit_descriptors.csv
```

These files contain compound identifiers, operational class labels, standardized SMILES strings, and calculated RDKit molecular descriptors used by the descriptor-based modeling pipeline.

---

## 5. Performance reporting

Training-set cross-validation metrics are reported as **mean ± sample standard deviation across ten folds**.

The held-out test set was evaluated separately after model selection based on the training-set analyses.

Test-set confidence intervals reported in the manuscript were estimated using **5000 stratified bootstrap resamples**.

Because the held-out test set was sampled from the same curated compound collection as the training set, the reported results represent **internal held-out validation rather than external validation**.

---

# Dataset construction

Drug–adverse-event associations were obtained from **SIDER version 4.1**, together with the literature-derived NCDIA dataset used in the previous study.

For model development, two operational class labels were defined.

## NCDIA-positive class

The NCDIA-positive class was constructed by combining:

1. drugs from the previously curated literature-derived NCDIA dataset; and
2. drugs annotated for agranulocytosis in SIDER.

## NCDIA-negative class

To construct the candidate NCDIA-negative class and reduce potential false-negative labels, a broad hematologic exclusion dictionary was applied.

Any SIDER drug annotated with at least one term related to:

- agranulocytosis;
- neutropenia;
- febrile neutropenia;
- leukopenia; or
- a quantitative or qualitative abnormality in white blood cell or neutrophil counts

was removed from the candidate NCDIA-negative pool.

These terms were used solely to remove potentially ambiguous compounds from the candidate NCDIA-negative pool and were not considered sufficient, by themselves, for assignment to the NCDIA-positive class.

Candidate NCDIA-negative drugs were additionally cross-checked against the literature sources used to construct the NCDIA-positive class.

When qualifying evidence of NCDIA was identified, the NCDIA-positive assignment took precedence.

Drugs with unresolved label conflicts or insufficient evidence were excluded from model development.

Chemotherapy agents classified under **Anatomical Therapeutic Chemical code L01** were removed to avoid confounding by expected dose-dependent myelosuppression.

After molecular standardization, filtering, and deduplication, the final dataset comprised **906 drugs**.

---

# Molecular representations

Four complementary open-source molecular representations were evaluated independently.

## RDKit descriptors

**208 RDKit descriptors** were calculated to characterize physicochemical, topological, electronic, and fragment-related molecular properties.

## Mordred descriptors

**1613 Mordred descriptors** were calculated, spanning constitutional, topological, autocorrelation, and electronic properties.

## MACCS keys

**166 binary MACCS structural keys** were generated to encode predefined structural patterns.

## ECFP4 fingerprints

**2048-bit extended-connectivity fingerprints** were generated using:

```text
Morgan radius = 2
diameter = 4
```

These fingerprints encode atom-centered circular environments extending up to two bonds from each atom.

---

# Feature preprocessing

The feature-preprocessing workflow included:

1. removal of features containing missing values;
2. z-score standardization of continuous RDKit and Mordred descriptors;
3. removal of zero-variance features; and
4. Pearson-correlation pruning, where applicable, using an absolute correlation threshold of `|r| > 0.9`.

Binary MACCS and ECFP4 features were not standardized.

---

# Feature selection

Five feature-selection strategies were evaluated.

## F-test

Analysis of variance was used to evaluate the univariate statistical association between each feature and the binary class label.

Features with:

```text
p < 0.05
```

were retained.

## Mutual information

Mutual information was used to capture linear and nonlinear statistical dependencies between individual features and the class label.

Features with:

```text
MI > 0
```

were retained.

## RFECV

Recursive feature elimination with cross-validation iteratively removed less informative features and selected a feature subset according to cross-validation performance.

## Genetic algorithm

The genetic algorithm evolved binary feature-selection masks using:

```text
population size                 = 50
generations                     = 40
tournament size                 = 3
crossover probability           = 0.5
per-individual mutation rate    = 0.2
per-gene bit-flip probability   = 0.05
```

## Embedded tree-based selection

Feature importance derived from a Balanced Random Forest model was used to identify informative features.

Features with importance:

```text
>= 0.001
```

were retained.

Balanced Random Forest was used as the base estimator for RFECV, GA, and embedded tree-based selection.

---

# Classifier benchmarking

For the selected RDKit_GA and MACCS_FP feature sets, five imbalance-aware ensemble classifiers were evaluated:

- **Balanced Random Forest (BRF)**
- **Easy Ensemble Classifier (EEC)**
- **Balanced Bagging + Gradient Boosting Decision Tree (BBC+GBDT)**
- **Balanced Bagging + LightGBM (BBC+LightGBM)**
- **Balanced Bagging + XGBoost (BBC+XGBoost)**

For comparison, standard and class-weighted versions of the following algorithms were also evaluated:

- Random Forest
- AdaBoost
- Gradient Boosting Decision Tree
- LightGBM
- XGBoost

This resulted in **15 classifiers per selected feature set**.

Hyperparameters were optimized using **Hyperopt version 0.2.7** with the tree-structured Parzen estimator.

Mean cross-validated **Matthews correlation coefficient (MCC)** was used as the primary optimization objective.

Final model selection was primarily based on cross-validated MCC. When MCC values were similar, sensitivity and the absolute sensitivity–specificity difference were used as secondary criteria.

---

# Performance metrics

Nine predictive-performance metrics were evaluated:

- receiver operating characteristic area under the curve (**ROC-AUC**);
- precision–recall area under the curve (**PR-AUC**);
- accuracy (**ACC**);
- balanced accuracy (**BA**);
- sensitivity (**SEN**);
- specificity (**SPE**);
- precision;
- F1 score; and
- Matthews correlation coefficient (**MCC**).

MCC was used as the principal model-selection metric because it incorporates all four outcomes of the confusion matrix and remains informative under class imbalance.

---

# Applicability domain

The applicability domain was assessed using the Euclidean-distance-based procedure described in the manuscript.

The analysis evaluates structural support relative to the corresponding training data.

For practical interpretation:

- an **in-domain** prediction indicates that the query compound lies within the predefined structural domain of the model;
- an **out-of-domain** prediction indicates extrapolation beyond that structural domain.

Applicability-domain membership should therefore be interpreted as an assessment of structural support rather than an absolute measure of prediction correctness or clinical risk.

---

# SHAP interpretation

SHAP was used to examine model behavior at both global and individual-compound levels.

## Global interpretation

Global SHAP analysis ranks molecular features according to their contributions to model outputs across the analyzed compounds.

The two final models provide complementary representation-specific information:

- **RDKit_GA_BRF** captures continuous physicochemical and topological molecular properties;
- **MACCS_FP_BRF** captures predefined structural motifs.

Because both models were developed using the same underlying dataset, agreement between their SHAP-derived patterns provides complementary cross-representation support but should not be interpreted as independent biological evidence.

## Local interpretation

For an individual compound, SHAP waterfall plots decompose the model output into feature-level contributions.

The waterfall plot helps identify which molecular properties or structural motifs shift the model output toward the NCDIA-positive or NCDIA-negative class.

## Important limitation of SHAP interpretation

SHAP explanations are conditional on the features available to the fitted model.

A descriptor or fingerprint key removed during preprocessing or feature selection cannot receive a SHAP attribution in that model. Its absence from a SHAP ranking should therefore not be interpreted as evidence of zero predictive or biological relevance.

More generally, SHAP describes associations within a fitted predictive model.

It does **not** demonstrate that:

- a highlighted descriptor causes agranulocytosis;
- a highlighted substructure is independently sufficient to cause NCDIA; or
- a proposed metabolic bioactivation pathway has been experimentally established for the query compound.

Mechanistic discussion should therefore be interpreted as literature-supported context for model-derived structural associations.

---

# Validation considerations

The study used stratified 10-fold cross-validation within the training set together with a held-out test set.

In the conventional training-set comparisons, preprocessing and feature selection were performed using the complete training set before the subsequent cross-validation comparisons. Therefore, the resulting conventional cross-validation estimates are conditional on the retained feature spaces and may contain some selection-induced optimism.

A supplementary stratified nested cross-validation analysis was also performed for the two selected final feature sets. In this analysis, BRF hyperparameter optimization was repeated within the inner folds, whereas the previously selected RDKit_GA and MACCS_FP feature sets were retained.

Accordingly, the supplementary nested analysis evaluates the robustness of hyperparameter optimization conditional on the selected feature spaces and should not be interpreted as fully end-to-end nesting of feature selection.

The held-out test set was not used for training-set cross-validation or hyperparameter optimization, but it was sampled from the same curated compound collection. External validation on independently assembled chemical and clinical datasets remains necessary.

---

# Scope and limitations

ExNCDIA is a structure-based predictive and interpretation framework. Several limitations should be considered when interpreting its outputs.

1. **NCDIA is multifactorial.**  
   Dose, pharmacokinetics, systemic exposure, treatment duration, concomitant medications, immune factors, genetic susceptibility, and other patient-level determinants cannot be represented by two-dimensional molecular structure alone.

2. **The dataset is relatively modest in size.**  
   The final curated collection contains 906 drugs.

3. **The class labels are operational.**  
   In particular, the NCDIA-negative class reflects the absence of qualifying evidence in the sources examined after the applied exclusion and cross-checking procedures rather than confirmed absence of NCDIA risk.

4. **Validation is internal.**  
   The held-out test set was sampled from the same curated compound collection as the training set. Generalization to new chemical space and real-world clinical populations requires future external and prospective evaluation.

5. **Conventional cross-validation is conditional on the selected feature spaces.**  
   Because feature preprocessing and selection were performed on the complete training set before the conventional cross-validation comparisons, some selection-induced optimism may remain.

6. **SHAP is associative rather than causal.**  
   The structural features identified by SHAP explain model behavior and should not be interpreted as experimentally proven causal mechanisms.

For these reasons, ExNCDIA should be used as a research, screening, and compound-prioritization tool rather than as a stand-alone clinical decision system.

---

# Citation

If you use the ExNCDIA code, models, web application, or dataset, please cite:

> Huang X, Jiang S, Liu P. *ExNCDIA: An Explainable Ensemble Learning Framework for Predicting Non-Chemotherapy Drug-Induced Agranulocytosis and Interpreting Structural Associations.* Manuscript; citation details to be updated upon publication.

---

# Funding

This work was supported by the **Medical Science and Technology Research Foundation of Guangdong Province** (Grant No. **A2024082**).

---

# Contact

**Xiaojie Huang**  
Department of Pharmacy  
Jieyang People's Hospital  
Jieyang 522000, China  
Email: huangxj46@alumni.sysu.edu.cn

---

# License

Released for academic and non-commercial research use. Please contact the corresponding author regarding other uses.
