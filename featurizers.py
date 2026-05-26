"""
featurizers.py
--------------
Turn a SMILES string into the numerical features each model needs.

  * standardize_mol      : clean / desalt / neutralize a molecule
                           (must match the standardization used at training time)
  * make_rdkit_calculator: build the 85-descriptor calculator
  * calc_rdkit_descriptors: molecule -> RDKit descriptor row
  * calc_maccs_bits      : molecule -> 167-bit MACCS fingerprint row
"""
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import pandas as pd


def standardize_mol(smiles: str):
    """Parse and standardize a SMILES string. Returns (mol, error_message)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES string"
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)         # remove salts / keep largest fragment
        mol = rdMolStandardize.Uncharger().uncharge(mol)   # neutralize charges
    except Exception as e:
        return None, f"Standardization failed: {e}"
    return mol, None


def make_rdkit_calculator(descriptor_names):
    """Build a descriptor calculator following the exact order stored in the model bundle."""
    return MolecularDescriptorCalculator(list(descriptor_names))


def calc_rdkit_descriptors(calc, mol, descriptor_names):
    """Molecule -> one-row DataFrame of RDKit descriptors, ordered as descriptor_names."""
    values = calc.CalcDescriptors(mol)
    return pd.DataFrame([dict(zip(descriptor_names, values))])[list(descriptor_names)]


def calc_maccs_bits(mol, n_bits: int = 167):
    """Molecule -> one-row DataFrame of MACCS bits, columns named MACCS_0 .. MACCS_166."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return pd.DataFrame([{f"MACCS_{i}": int(fp.GetBit(i)) for i in range(n_bits)}])
