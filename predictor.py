"""
predictor.py
------------
Feature building and prediction for the two ExNCDIA sub-models
(the RDKit-descriptor model and the MACCS-fingerprint model).
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from featurizers import calc_rdkit_descriptors, calc_maccs_bits


def build_features(bundle, mol, rdkit_calc=None):
    """Build the model input for one molecule.

    Returns (raw_df, Xq):
      raw_df : 1-row DataFrame of human-readable feature values for the
               model's features - un-standardized RDKit descriptors, or
               0/1 MACCS bits. Used for display, SHAP and download.
      Xq     : numpy array the model actually consumes
               (z-score standardized for the RDKit model).
    """
    kind = bundle["meta"]["descriptor_type"]
    if kind == "RDKit":
        raw_full = calc_rdkit_descriptors(rdkit_calc, mol, bundle["descriptor_85"])
        if not np.all(np.isfinite(raw_full.values)):
            raise ValueError("Descriptor vector contains NaN / Inf")
        std = pd.DataFrame(bundle["scaler"].transform(raw_full),
                           columns=bundle["descriptor_85"])
        raw_df = raw_full[bundle["final_features"]]
        Xq = std[bundle["final_features"]].values
    elif kind == "MACCS":
        raw_full = calc_maccs_bits(mol, bundle["meta"]["n_maccs_bits"])
        raw_df = raw_full[bundle["final_features"]]
        Xq = raw_df.values
    else:
        raise ValueError(f"Unknown model type: {kind}")
    return raw_df, Xq


def _ad_status(Xq, bundle):
    """Applicability-domain check (Euclidean for RDKit, Tanimoto for MACCS)."""
    metric = bundle.get("ad_metric", "euclidean")
    ref = bundle["ad_train_matrix"]
    q = Xq.astype(bool) if metric == "jaccard" else Xq
    d = cdist(q, ref, metric=metric).mean(axis=1)[0]
    d_norm = (d - bundle["ad_d_min"]) / (bundle["ad_d_max"] - bundle["ad_d_min"])
    return float(d_norm), bool(d_norm <= bundle["ad_threshold"])


def predict_from_Xq(bundle, Xq):
    """Predict from an already-prepared model-input array."""
    model = bundle["model"]
    # Wrap in a DataFrame with the training feature names (avoids the
    # sklearn "X does not have valid feature names" warning).
    Xdf = pd.DataFrame(np.asarray(Xq), columns=bundle["final_features"])
    pos_idx = list(model.classes_).index(bundle["meta"]["positive_label"])
    proba = float(model.predict_proba(Xdf)[0, pos_idx])
    label = int(model.predict(Xdf)[0])
    d_norm, in_ad = _ad_status(np.asarray(Xq), bundle)
    return {"label": label, "proba": proba, "ad_norm": d_norm, "in_ad": in_ad}


def predict(bundle, mol, rdkit_calc=None):
    """Convenience: featurize + predict for one molecule."""
    _, Xq = build_features(bundle, mol, rdkit_calc)
    return predict_from_Xq(bundle, Xq)
