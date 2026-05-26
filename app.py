"""
app.py
------
Streamlit web app for the NCDIA risk predictor.

Enter a SMILES string and get predictions from two independently trained
Balanced Random Forest models (RDKit descriptors & MACCS fingerprints),
each shown with its applicability-domain assessment.

Run locally:   streamlit run app.py
"""
import streamlit as st
import joblib
from rdkit.Chem import Draw
from featurizers import standardize_mol, make_rdkit_calculator
from predictor import predict

st.set_page_config(page_title="NCDIA Risk Predictor", layout="centered")


@st.cache_resource
def load_models():
    """Load both model bundles once and reuse across reruns."""
    rdkit_bundle = joblib.load("NCDIA_RDKit_BRF_model.pkl")
    maccs_bundle = joblib.load("NCDIA_MACCS_BRF_model.pkl")
    rdkit_calc = make_rdkit_calculator(rdkit_bundle["descriptor_85"])
    return rdkit_bundle, maccs_bundle, rdkit_calc


rdkit_bundle, maccs_bundle, rdkit_calc = load_models()

st.title("NCDIA Risk Predictor")
st.caption("Non-chemotherapy Drug-Induced Agranulocytosis — Balanced Random "
           "Forest models (RDKit descriptors & MACCS fingerprints)")

smiles = st.text_input("Enter a compound SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")

if st.button("Predict", type="primary"):
    mol, err = standardize_mol(smiles)
    if err:
        st.error(err)
        st.stop()

    st.image(Draw.MolToImage(mol, size=(300, 240)), caption="Standardized structure")

    cols = st.columns(2)
    for col, bundle, name in [(cols[0], rdkit_bundle, "RDKit Descriptor Model"),
                              (cols[1], maccs_bundle, "MACCS Fingerprint Model")]:
        with col:
            st.subheader(name)
            try:
                r = predict(bundle, mol, rdkit_calc)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                continue
            st.metric("Prediction",
                      "High risk (positive)" if r["label"] == 1 else "Low risk (negative)")
            st.metric("Probability of positive", f"{r['proba']:.1%}")
            st.metric("Applicability domain",
                      "Inside" if r["in_ad"] else "Outside")
            if not r["in_ad"]:
                st.warning(f"Normalized distance {r['ad_norm']:.2f} > 1.0. This "
                           f"compound lies outside the model's applicability domain; "
                           f"interpret the prediction with caution.")

    st.divider()
    st.caption("The two models were trained independently on different molecular "
               "representations. When they disagree, give more weight to the model "
               "whose applicability-domain status is 'Inside'.")
