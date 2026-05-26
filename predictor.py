"""
predictor.py
------------
Unified prediction logic for both the RDKit-descriptor model and the
MACCS-fingerprint model.

For a given model bundle (.pkl) and a molecule it:
  1. builds the raw features
  2. (RDKit only) standardizes them with the saved scaler
  3. selects the columns the model was trained on
  4. predicts class label and probability
  5. evaluates the applicability domain (AD)
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from featurizers import calc_rdkit_descriptors, calc_maccs_bits


def _ad_status(Xq, bundle):
    """Applicability-domain check.

    RDKit model -> Euclidean distance on standardized descriptors.
    MACCS model -> Tanimoto distance (scipy 'jaccard' = 1 - Tanimoto similarity).
    """
    metric = bundle.get("ad_metric", "euclidean")
    ref = bundle["ad_train_matrix"]
    q = Xq.astype(bool) if metric == "jaccard" else Xq
    d = cdist(q, ref, metric=metric).mean(axis=1)[0]
    d_norm = (d - bundle["ad_d_min"]) / (bundle["ad_d_max"] - bundle["ad_d_min"])
    return float(d_norm), bool(d_norm <= bundle["ad_threshold"])


def predict(bundle, mol, rdkit_calc=None):
    """Predict NCDIA risk for a single molecule with the given model bundle.

    Returns a dict: {label, proba, ad_norm, in_ad}.
    """
    kind = bundle["meta"]["descriptor_type"]

    if kind == "RDKit":
        raw = calc_rdkit_descriptors(rdkit_calc, mol, bundle["descriptor_85"])
        if not np.all(np.isfinite(raw.values)):
            raise ValueError("Descriptor vector contains NaN / Inf")
        std = pd.DataFrame(bundle["scaler"].transform(raw),
                           columns=bundle["descriptor_85"])
        Xq = std[bundle["final_features"]].values
    elif kind == "MACCS":
        raw = calc_maccs_bits(mol, bundle["meta"]["n_maccs_bits"])
        Xq = raw[bundle["final_features"]].values
    else:
        raise ValueError(f"Unknown model type: {kind}")

    model = bundle["model"]
    pos_idx = list(model.classes_).index(bundle["meta"]["positive_label"])
    proba = float(model.predict_proba(Xq)[0, pos_idx])
    label = int(model.predict(Xq)[0])
    d_norm, in_ad = _ad_status(Xq, bundle)
    return {"label": label, "proba": proba, "ad_norm": d_norm, "in_ad": in_ad}
