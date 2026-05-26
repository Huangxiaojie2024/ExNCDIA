"""
app.py
------
ExNCDIA - Explainable predictor of non-chemotherapy drug-induced
agranulocytosis (NCDIA).

Two Balanced Random Forest models (RDKit descriptors & MACCS fingerprints),
each with an applicability-domain check and SHAP-based explanation.

Run locally:  streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from rdkit import Chem

from featurizers import standardize_mol, make_rdkit_calculator
from predictor import build_features, predict_from_Xq
from explainer import (build_explainer, shap_for_molecule,
                       waterfall_figure, top_feature_table)

# Molecule drawing is optional (needs system X11 libs - see packages.txt).
try:
    from rdkit.Chem import Draw
    DRAW_OK = True
except Exception:
    DRAW_OK = False

st.set_page_config(page_title="ExNCDIA", layout="wide")

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  .hero {
      background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
      border-radius: 16px; padding: 26px 32px; color: #fff; margin-bottom: 8px;
  }
  .hero-title {font-size: 2.5rem; font-weight: 800; letter-spacing: .5px;}
  .hero-tag {font-size: 1.05rem; opacity: .95; margin-top: 2px;}
  .hero-meta {font-size: .85rem; opacity: .8; margin-top: 10px;}
  .card {
      background: #fff; border: 1px solid #E5E8F0; border-radius: 14px;
      padding: 18px 20px; box-shadow: 0 1px 3px rgba(20,30,60,.06);
      height: 100%;
  }
  .card-title {font-size: 1.1rem; font-weight: 700; color: #1A1F36;}
  .card-sub {font-size: .8rem; color: #6B7280; margin-bottom: 12px;}
  .badge {
      display: inline-block; padding: 5px 14px; border-radius: 999px;
      font-weight: 700; font-size: .92rem; color: #fff; margin-bottom: 12px;
  }
  .badge.high {background: #E64B35;}
  .badge.low  {background: #2E9E5B;}
  .prob-row {display: flex; justify-content: space-between; align-items: baseline;}
  .prob-label {font-size: .82rem; color: #6B7280;}
  .prob-val {font-size: 1.6rem; font-weight: 800; color: #1A1F36;}
  .bar {background: #EEF1F6; border-radius: 999px; height: 9px; margin: 6px 0 12px 0;}
  .bar-fill {height: 9px; border-radius: 999px;}
  .bar-fill.high {background: #E64B35;}
  .bar-fill.low  {background: #2E9E5B;}
  .chip {
      display: inline-block; padding: 3px 11px; border-radius: 999px;
      font-size: .78rem; font-weight: 600;
  }
  .chip.in  {background: #E8F3FF; color: #1D4ED8;}
  .chip.out {background: #FDEEEA; color: #C0392B;}
  .consensus {
      background: #F4F6FA; border-left: 4px solid #2563EB; border-radius: 8px;
      padding: 12px 16px; font-size: .92rem; color: #1A1F36; margin-top: 4px;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-title">ExNCDIA</div>
  <div class="hero-tag">Explainable prediction of non-chemotherapy drug-induced agranulocytosis</div>
  <div class="hero-meta">Dual Balanced Random Forest models &nbsp;&middot;&nbsp;
     RDKit descriptors + MACCS fingerprints &nbsp;&middot;&nbsp; SHAP-explained &nbsp;&middot;&nbsp;
     applicability-domain aware</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Resources (loaded once)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and preparing SHAP explainers...")
def load_resources():
    rb = joblib.load("NCDIA_RDKit_BRF_model.pkl")
    mb = joblib.load("NCDIA_MACCS_BRF_model.pkl")
    rcalc = make_rdkit_calculator(rb["descriptor_85"])
    rexpl = build_explainer(rb)
    mexpl = build_explainer(mb)
    return dict(rb=rb, mb=mb, rcalc=rcalc, rexpl=rexpl, mexpl=mexpl)


R = load_resources()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def card_html(name, subtitle, res):
    high = res["label"] == 1
    risk_txt = "HIGH RISK" if high else "LOW RISK"
    cls = "high" if high else "low"
    ad_txt = "Inside applicability domain" if res["in_ad"] else "Outside applicability domain"
    ad_cls = "in" if res["in_ad"] else "out"
    pct = res["proba"] * 100
    return f"""
    <div class="card">
      <div class="card-title">{name}</div>
      <div class="card-sub">{subtitle}</div>
      <span class="badge {cls}">{risk_txt}</span>
      <div class="prob-row">
        <span class="prob-label">Probability of NCDIA-positive</span>
        <span class="prob-val">{pct:.1f}%</span>
      </div>
      <div class="bar"><div class="bar-fill {cls}" style="width:{pct:.1f}%"></div></div>
      <span class="chip {ad_cls}">{ad_txt} &nbsp;(d = {res['ad_norm']:.2f})</span>
    </div>
    """


def consensus_text(rr, rm):
    if rr["label"] == rm["label"]:
        verdict = "HIGH-RISK" if rr["label"] == 1 else "LOW-RISK"
        msg = f"Both models agree: <b>{verdict}</b> for NCDIA."
    else:
        msg = ("The two models <b>disagree</b>. Give more weight to the model whose "
               "result is inside the applicability domain; if both are inside, treat "
               "the compound as borderline and review it manually.")
    if not (rr["in_ad"] and rm["in_ad"]):
        msg += " &nbsp;Note: at least one model flags this compound as outside its applicability domain."
    return msg


def parse_smiles_input(text, uploaded):
    items = []
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(raw))
            col = next((c for c in df.columns if c.strip().lower() in ("smiles", "smile")),
                       df.columns[0])
            items = df[col].astype(str).tolist()
        else:
            items = [ln.strip() for ln in raw.splitlines()]
    if text and text.strip():
        items += [ln.strip() for ln in text.splitlines()]
    return [s for s in items if s and s.lower() not in ("smiles", "smile")]


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_single, tab_batch, tab_about = st.tabs(
    ["  Single prediction  ", "  Batch prediction  ", "  About  "])

# ===================== SINGLE =====================
with tab_single:
    st.markdown("##### Enter a compound")
    smiles = st.text_input("SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O",
                           label_visibility="collapsed")
    if st.button("Predict", type="primary"):
        mol, err = standardize_mol(smiles)
        if err:
            st.error(err)
            st.session_state.pop("pred", None)
        else:
            try:
                raw_r, Xq_r = build_features(R["rb"], mol, R["rcalc"])
                res_r = predict_from_Xq(R["rb"], Xq_r)
                raw_m, Xq_m = build_features(R["mb"], mol, None)
                res_m = predict_from_Xq(R["mb"], Xq_m)
                st.session_state["pred"] = dict(
                    smiles=smiles, std=Chem.MolToSmiles(mol), mol=mol,
                    rdkit=dict(raw=raw_r, Xq=Xq_r, res=res_r),
                    maccs=dict(raw=raw_m, Xq=Xq_m, res=res_m))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.pop("pred", None)

    if "pred" in st.session_state:
        p = st.session_state["pred"]
        c1, c2, c3 = st.columns([1.0, 1.25, 1.25])
        with c1:
            if DRAW_OK:
                try:
                    st.image(Draw.MolToImage(p["mol"], size=(260, 230)))
                except Exception:
                    pass
            st.caption("Standardized SMILES")
            st.code(p["std"], language=None)
        with c2:
            st.markdown(card_html("Descriptor Model", "RDKit \u00b7 85 descriptors",
                                  p["rdkit"]["res"]), unsafe_allow_html=True)
        with c3:
            st.markdown(card_html("Fingerprint Model", "MACCS \u00b7 131 keys",
                                  p["maccs"]["res"]), unsafe_allow_html=True)

        st.markdown(f'<div class="consensus">{consensus_text(p["rdkit"]["res"], p["maccs"]["res"])}</div>',
                    unsafe_allow_html=True)

        # ----- SHAP explanation -----
        st.markdown("---")
        st.markdown("##### Why this prediction? &nbsp; SHAP explanation")
        which = st.radio("Explain with", ["Descriptor Model (RDKit)", "Fingerprint Model (MACCS)"],
                         horizontal=True, label_visibility="collapsed")
        if which.startswith("Descriptor"):
            expl, sub = R["rexpl"], p["rdkit"]
        else:
            expl, sub = R["mexpl"], p["maccs"]

        shap_vals, base = shap_for_molecule(expl, sub["Xq"])
        fnames = list(sub["raw"].columns)
        fvals = sub["raw"].values[0]

        st.markdown("**This molecule &mdash; top-5 driving features**")
        fig = waterfall_figure(shap_vals, base, fnames, fvals, max_display=6)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red bars push the predicted probability up (toward NCDIA-positive), "
                   "blue bars push it down. Ranking is specific to THIS compound.")

        st.dataframe(pd.DataFrame(top_feature_table(shap_vals, fnames, fvals, k=5)),
                     hide_index=True, use_container_width=True)

        # ----- Descriptor download -----
        st.markdown("---")
        feat = {"SMILES": p["std"]}
        for c in p["rdkit"]["raw"].columns:
            feat[f"RDKit::{c}"] = float(p["rdkit"]["raw"][c].values[0])
        for c in p["maccs"]["raw"].columns:
            feat[c] = int(p["maccs"]["raw"][c].values[0])
        feat_csv = pd.DataFrame([feat]).to_csv(index=False)
        st.download_button("Download computed descriptors & fingerprints (CSV)",
                           feat_csv, "ExNCDIA_features.csv", "text/csv")

# ===================== BATCH =====================
with tab_batch:
    st.markdown("##### Predict many compounds")
    st.caption("Paste SMILES (one per line) and/or upload a .csv (with a 'SMILES' column) "
               "or .txt file. Up to 300 compounds per run.")
    txt = st.text_area("SMILES list", height=130, label_visibility="collapsed",
                       placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    up = st.file_uploader("Upload .csv / .txt", type=["csv", "txt"],
                          label_visibility="collapsed")

    if st.button("Run batch prediction", type="primary"):
        smiles_list = parse_smiles_input(txt, up)
        if not smiles_list:
            st.warning("No SMILES provided.")
        else:
            if len(smiles_list) > 300:
                st.warning(f"{len(smiles_list)} inputs received; only the first 300 are processed.")
                smiles_list = smiles_list[:300]
            rows, prog = [], st.progress(0.0)
            for i, smi in enumerate(smiles_list):
                mol, err = standardize_mol(smi)
                if err:
                    rows.append({"SMILES": smi, "Status": err})
                else:
                    try:
                        _, xqr = build_features(R["rb"], mol, R["rcalc"])
                        rr = predict_from_Xq(R["rb"], xqr)
                        _, xqm = build_features(R["mb"], mol, None)
                        rm = predict_from_Xq(R["mb"], xqm)
                        agree = rr["label"] == rm["label"]
                        rows.append({
                            "SMILES": smi,
                            "Standardized": Chem.MolToSmiles(mol),
                            "Status": "OK",
                            "RDKit_pred": "Positive" if rr["label"] else "Negative",
                            "RDKit_prob": round(rr["proba"], 3),
                            "RDKit_AD": "In" if rr["in_ad"] else "Out",
                            "MACCS_pred": "Positive" if rm["label"] else "Negative",
                            "MACCS_prob": round(rm["proba"], 3),
                            "MACCS_AD": "In" if rm["in_ad"] else "Out",
                            "Consensus": ("Positive" if (agree and rr["label"]) else
                                          "Negative" if agree else "Disagree"),
                        })
                    except Exception as e:
                        rows.append({"SMILES": smi, "Status": f"Failed: {e}"})
                prog.progress((i + 1) / len(smiles_list))
            st.session_state["batch"] = pd.DataFrame(rows)

    if "batch" in st.session_state:
        df = st.session_state["batch"]
        ok = (df.get("Status") == "OK").sum() if "Status" in df else 0
        st.markdown(f"**{len(df)} compounds processed &middot; {ok} succeeded**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", df.to_csv(index=False),
                           "ExNCDIA_batch_results.csv", "text/csv")

# ===================== ABOUT =====================
with tab_about:
    st.markdown("""
##### About ExNCDIA

**ExNCDIA** is an explainable machine-learning tool for predicting **non-chemotherapy
drug-induced agranulocytosis (NCDIA)** &mdash; a rare but potentially fatal hematological
adverse reaction marked by a severe reduction in circulating neutrophils, which can
progress rapidly to serious infection and sepsis.

Unlike chemotherapy-induced neutropenia, which results mainly from dose-dependent bone
marrow suppression, NCDIA arises through heterogeneous and idiosyncratic mechanisms
&mdash; including immune-mediated neutrophil destruction and direct toxicity toward
myeloid precursors &mdash; making it difficult to anticipate from chemical structure
alone. ExNCDIA was built to support **early structural risk assessment** in medicinal
chemistry and drug-safety workflows.

##### How it works

ExNCDIA combines **two independently trained Balanced Random Forest models**:

- **Descriptor Model** &mdash; 85 RDKit molecular descriptors (genetic-algorithm selected),
  capturing physicochemical properties and topology.
- **Fingerprint Model** &mdash; 131 MACCS structural keys, capturing substructural motifs.

Both were developed on **906 curated compounds** (371 NCDIA-positive, 535 negative),
split into a training set of 724 and an independent test set of 182. Every prediction
is accompanied by a **SHAP explanation** identifying the molecular features that drive
that specific compound's outcome.

##### Reading the output

- **Risk badge & probability** &mdash; the model's class call and the estimated
  probability of being NCDIA-positive.
- **Applicability domain (AD)** &mdash; whether the compound is structurally similar
  enough to the training data for the prediction to be reliable. A result *outside*
  the AD should be interpreted with caution.
- **SHAP waterfall** &mdash; the features pushing this molecule's prediction up (red)
  or down (blue), ranked for this compound; a model-wide ranking is shown alongside
  for comparison.

##### Citation & disclaimer

Huang X. *ExNCDIA: Explainable prediction and mechanistic insights into
non-chemotherapy drug-induced agranulocytosis through ensemble machine learning
approaches.* Department of Pharmacy, Jieyang People's Hospital.

ExNCDIA is provided for **research and educational purposes only**. It is not a
medical device and does not replace experimental toxicity assessment or clinical
judgement.
""")
