"""
explainer.py
------------
SHAP explanations for the ExNCDIA models.

Provides per-molecule explanations (which features drive THIS compound's
prediction) and a model-wide importance ranking for comparison.

For the RDKit model, SHAP values are computed on the standardized model
input, but feature VALUES displayed to the user are the original
(un-standardized) descriptor values - handled by passing `raw_df` from
predictor.build_features as the display data.
"""
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_explainer(bundle):
    """Create a TreeExplainer for a model bundle."""
    return shap.TreeExplainer(bundle["model"])


def _class1(raw_sv, explainer):
    """Extract class-1 SHAP values and base value from TreeExplainer output."""
    sv = np.asarray(raw_sv)
    base = float(np.ravel(explainer.expected_value)[-1])
    if sv.ndim == 3:                       # (n_samples, n_features, n_classes)
        return sv[:, :, -1], base
    return sv, base                        # already 2-D


def shap_for_molecule(explainer, Xq):
    """Per-molecule class-1 SHAP values.

    Xq : 1-row model-input array. Returns (shap_values[n_features], base).
    """
    sv, base = _class1(explainer.shap_values(Xq), explainer)
    return sv[0], base


def global_shap_importance(bundle, explainer):
    """Model-wide importance = mean |SHAP| over the training set.

    Returns a 1-D array aligned with bundle['final_features'].
    """
    Xt = np.asarray(bundle["ad_train_matrix"], dtype=float)
    sv, _ = _class1(explainer.shap_values(Xt), explainer)
    return np.abs(sv).mean(axis=0)


def waterfall_figure(shap_vals, base, feature_names, feature_values,
                     max_display=6):
    """SHAP waterfall for a single molecule.

    Features are ranked by |SHAP| FOR THIS MOLECULE; the most important
    (max_display - 1) are shown individually and the rest are merged into
    one 'other features' row.
    """
    expl = shap.Explanation(
        values=np.asarray(shap_vals, dtype=float),
        base_values=float(base),
        data=np.asarray(feature_values, dtype=float),
        feature_names=list(feature_names),
    )
    plt.close("all")
    shap.plots.waterfall(expl, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.4, 4.8)
    plt.tight_layout()
    return fig


def top_feature_table(shap_vals, feature_names, feature_values, k=5):
    """Top-k features for THIS molecule, ranked by |SHAP|."""
    shap_vals = np.asarray(shap_vals, dtype=float)
    order = np.argsort(np.abs(shap_vals))[::-1][:k]
    rows = []
    for rank, i in enumerate(order, start=1):
        v = float(feature_values[i])
        v_disp = f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.3f}"
        rows.append({
            "Rank": rank,
            "Feature": feature_names[i],
            "Value": v_disp,
            "SHAP": round(float(shap_vals[i]), 4),
            "Effect": "increases risk" if shap_vals[i] >= 0 else "decreases risk",
        })
    return rows


def global_bar_figure(importance, feature_names, k=5):
    """Horizontal bar chart of the model-wide top-k features (mean |SHAP|)."""
    importance = np.asarray(importance, dtype=float)
    order = np.argsort(importance)[::-1][:k]
    names = [feature_names[i] for i in order][::-1]
    vals = [importance[i] for i in order][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.barh(names, vals, color="#2563EB", height=0.6)
    ax.set_xlabel("Mean |SHAP|  (training set)", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig
