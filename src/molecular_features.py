"""
molecular_features.py
=====================
RDKit-based molecular descriptor extraction and fingerprint generation.
Produces 200+ features per molecule.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, AllChem, MACCSkeys,
    rdFingerprintGenerator, Lipinski, GraphDescriptors,
    Fragments, QED
)
from rdkit.Chem import FilterCatalog
from rdkit import DataStructs

logger = logging.getLogger(__name__)

# ─── Descriptor list ────────────────────────────────────────────────────────

DESCRIPTOR_FUNCTIONS = {
    # Lipinski / Druglikeness
    "MolWt":            Descriptors.MolWt,
    "ExactMolWt":       Descriptors.ExactMolWt,
    "LogP":             Descriptors.MolLogP,
    "TPSA":             Descriptors.TPSA,
    "HBD":              rdMolDescriptors.CalcNumHBD,
    "HBA":              rdMolDescriptors.CalcNumHBA,
    "RotBonds":         rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount":        rdMolDescriptors.CalcNumRings,
    "AromaticRings":    rdMolDescriptors.CalcNumAromaticRings,
    "HeavyAtomCount":   Descriptors.HeavyAtomCount,
    "FractionCSP3":     rdMolDescriptors.CalcFractionCSP3,
    "NumAliphaticRings":rdMolDescriptors.CalcNumAliphaticRings,
    "NumSaturatedRings":rdMolDescriptors.CalcNumSaturatedRings,
    "NumHeteroatoms":   rdMolDescriptors.CalcNumHeteroatoms,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    # Electronic
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
    "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
    # Topological
    "BertzCT":          Descriptors.BertzCT,
    "Chi0":             Descriptors.Chi0,
    "Chi0n":            Descriptors.Chi0n,
    "Chi0v":            Descriptors.Chi0v,
    "Chi1":             Descriptors.Chi1,
    "Chi1n":            Descriptors.Chi1n,
    "Chi1v":            Descriptors.Chi1v,
    "Chi2n":            Descriptors.Chi2n,
    "Chi2v":            Descriptors.Chi2v,
    "Chi3n":            Descriptors.Chi3n,
    "Chi3v":            Descriptors.Chi3v,
    "Chi4n":            Descriptors.Chi4n,
    "Chi4v":            Descriptors.Chi4v,
    "HallKierAlpha":    Descriptors.HallKierAlpha,
    "Kappa1":           Descriptors.Kappa1,
    "Kappa2":           Descriptors.Kappa2,
    "Kappa3":           Descriptors.Kappa3,
    # VSA
    "LabuteASA":        Descriptors.LabuteASA,
    "PEOE_VSA1":        Descriptors.PEOE_VSA1,
    "PEOE_VSA2":        Descriptors.PEOE_VSA2,
    "PEOE_VSA3":        Descriptors.PEOE_VSA3,
    "SMR_VSA1":         Descriptors.SMR_VSA1,
    "SMR_VSA2":         Descriptors.SMR_VSA2,
    "SMR_VSA3":         Descriptors.SMR_VSA3,
    "SlogP_VSA1":       Descriptors.SlogP_VSA1,
    "SlogP_VSA2":       Descriptors.SlogP_VSA2,
    "SlogP_VSA3":       Descriptors.SlogP_VSA3,
    # Estate
    "MaxEStateIndex":   Descriptors.MaxEStateIndex,
    "MinEStateIndex":   Descriptors.MinEStateIndex,
    "MaxAbsEStateIndex":Descriptors.MaxAbsEStateIndex,
    "MinAbsEStateIndex":Descriptors.MinAbsEStateIndex,
    # QED
    "QED":              lambda m: QED.qed(m),
    # Fragments (functional groups)
    "fr_Al_COO":        Fragments.fr_Al_COO,
    "fr_Al_OH":         Fragments.fr_Al_OH,
    "fr_ArN":           Fragments.fr_ArN,
    "fr_Ar_OH":         Fragments.fr_Ar_OH,
    "fr_COO":           Fragments.fr_COO,
    "fr_COO2":          Fragments.fr_COO2,
    "fr_C_O":           Fragments.fr_C_O,
    "fr_C_S":           Fragments.fr_C_S,
    "fr_HOCCN":         Fragments.fr_HOCCN,
    "fr_Imine":         Fragments.fr_Imine,
    "fr_NH0":           Fragments.fr_NH0,
    "fr_NH1":           Fragments.fr_NH1,
    "fr_NH2":           Fragments.fr_NH2,
    "fr_N_O":           Fragments.fr_N_O,
    "fr_Ndealkylation1":Fragments.fr_Ndealkylation1,
    "fr_Ndealkylation2":Fragments.fr_Ndealkylation2,
    "fr_Nhpyrrole":     Fragments.fr_Nhpyrrole,
    "fr_SH":            Fragments.fr_SH,
    "fr_aldehyde":      Fragments.fr_aldehyde,
    "fr_alkyl_carbamate": Fragments.fr_alkyl_carbamate,
    "fr_alkyl_halide":  Fragments.fr_alkyl_halide,
    "fr_allylic_oxid":  Fragments.fr_allylic_oxid,
    "fr_amide":         Fragments.fr_amide,
    "fr_amidine":       Fragments.fr_amidine,
    "fr_aniline":       Fragments.fr_aniline,
    "fr_aryl_methyl":   Fragments.fr_aryl_methyl,
    "fr_azide":         Fragments.fr_azide,
    "fr_azo":           Fragments.fr_azo,
    "fr_barbitur":      Fragments.fr_barbitur,
    "fr_benzimidazole": Fragments.fr_imidazole,
    "fr_benzodiazepine":Fragments.fr_imide,
    "fr_bicyclic":      Fragments.fr_bicyclic,
    "fr_diazo":         Fragments.fr_diazo,
    "fr_dihydropyridine":Fragments.fr_dihydropyridine,
    "fr_epoxide":       Fragments.fr_epoxide,
    "fr_ester":         Fragments.fr_ester,
    "fr_ether":         Fragments.fr_ether,
    "fr_furan":         Fragments.fr_furan,
    "fr_guanido":       Fragments.fr_guanido,
    "fr_halogen":       Fragments.fr_halogen,
    "fr_hdrzine":       Fragments.fr_hdrzine,
    "fr_hdrzone":       Fragments.fr_hdrzone,
    "fr_imidazole":     Fragments.fr_imidazole,
    "fr_imide":         Fragments.fr_imide,
    "fr_isocyan":       Fragments.fr_isocyan,
    "fr_isothiocyan":   Fragments.fr_isothiocyan,
    "fr_ketone":        Fragments.fr_ketone,
    "fr_ketone_Topliss":Fragments.fr_ketone_Topliss,
    "fr_lactam":        Fragments.fr_lactam,
    "fr_lactone":       Fragments.fr_lactone,
    "fr_methoxy":       Fragments.fr_methoxy,
    "fr_morpholine":    Fragments.fr_morpholine,
    "fr_nitrile":       Fragments.fr_nitrile,
    "fr_nitro":         Fragments.fr_nitro,
    "fr_nitro_arom":    Fragments.fr_nitro_arom,
    "fr_nitroso":       Fragments.fr_nitroso,
    "fr_oxazole":       Fragments.fr_oxazole,
    "fr_oxime":         Fragments.fr_oxime,
    "fr_para_hydroxylation": Fragments.fr_para_hydroxylation,
    "fr_phenol":        Fragments.fr_phenol,
    "fr_phenol_noOrthoHbond": Fragments.fr_phenol_noOrthoHbond,
    "fr_phos_acid":     Fragments.fr_phos_acid,
    "fr_phos_ester":    Fragments.fr_phos_ester,
    "fr_piperdine":     Fragments.fr_piperdine,
    "fr_piperzine":     Fragments.fr_piperzine,
    "fr_priamide":      Fragments.fr_priamide,
    "fr_pyridine":      Fragments.fr_pyridine,
    "fr_quatN":         Fragments.fr_quatN,
    "fr_sulfide":       Fragments.fr_sulfide,
    "fr_sulfonamd":     Fragments.fr_sulfonamd,
    "fr_sulfone":       Fragments.fr_sulfone,
    "fr_term_acetylene":Fragments.fr_term_acetylene,
    "fr_tetrazole":     Fragments.fr_tetrazole,
    "fr_thiazole":      Fragments.fr_thiazole,
    "fr_thiocyan":      Fragments.fr_thiocyan,
    "fr_thiophene":     Fragments.fr_thiophene,
    "fr_unbrch_alkane": Fragments.fr_unbrch_alkane,
    "fr_urea":          Fragments.fr_urea,
}


def smiles_to_mol(smiles: str) -> Optional[object]:
    """Convert SMILES string to RDKit Mol. Returns None if invalid."""
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol  # None if invalid
    except Exception:
        return None


def compute_descriptors(mol) -> dict:
    """Compute all molecular descriptors for a single RDKit mol object."""
    result = {}
    for name, fn in DESCRIPTOR_FUNCTIONS.items():
        try:
            val = fn(mol)
            result[name] = float(val) if val is not None else np.nan
        except Exception:
            result[name] = np.nan
    return result


def morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Generate Morgan (ECFP4) fingerprint as numpy array."""
    try:
        import warnings
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.warning')
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.int8)


def maccs_fingerprint(mol) -> np.ndarray:
    """Generate MACCS Keys fingerprint (167 bits)."""
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(167, dtype=np.int8)


def rdkit_fingerprint(mol, n_bits: int = 2048) -> np.ndarray:
    """Generate RDKit topological fingerprint."""
    try:
        from rdkit.Chem import RDKFingerprint
        fp = RDKFingerprint(mol, fpSize=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.int8)


def process_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    task_cols: list = None,
    n_bits: int = 2048,
) -> tuple:
    """
    Full pipeline: SMILES → descriptors + fingerprints.

    Returns:
        desc_df     : DataFrame of molecular descriptors (valid molecules only)
        morgan_arr  : ndarray of Morgan fingerprints
        maccs_arr   : ndarray of MACCS keys
        rdkit_arr   : ndarray of RDKit fingerprints
        valid_idx   : boolean mask of valid molecules
    """
    logger.info(f"Processing {len(df):,} SMILES strings...")

    mols = []
    valid_mask = []
    for smi in df[smiles_col]:
        mol = smiles_to_mol(smi)
        if mol is not None:
            mols.append(mol)
            valid_mask.append(True)
        else:
            mols.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask, dtype=bool)
    valid_mols = [m for m in mols if m is not None]
    n_valid = len(valid_mols)
    n_invalid = valid_mask.shape[0] - n_valid
    logger.info(f"Valid molecules: {n_valid:,} | Invalid/skipped: {n_invalid:,}")

    # Descriptors
    logger.info("Computing molecular descriptors...")
    desc_list = [compute_descriptors(m) for m in valid_mols]
    desc_df = pd.DataFrame(desc_list)
    logger.info(f"Descriptor matrix: {desc_df.shape}")

    # Morgan fingerprints
    logger.info("Generating Morgan fingerprints (ECFP4, 2048 bits)...")
    morgan_arr = np.vstack([morgan_fingerprint(m, radius=2, n_bits=n_bits) for m in valid_mols])

    # MACCS keys
    logger.info("Generating MACCS Keys (167 bits)...")
    maccs_arr = np.vstack([maccs_fingerprint(m) for m in valid_mols])

    # RDKit fingerprints
    logger.info("Generating RDKit topological fingerprints (2048 bits)...")
    rdkit_arr = np.vstack([rdkit_fingerprint(m, n_bits=n_bits) for m in valid_mols])

    logger.info("Feature generation complete.")
    logger.info(f"  Descriptors : {desc_df.shape}")
    logger.info(f"  Morgan FP   : {morgan_arr.shape}")
    logger.info(f"  MACCS Keys  : {maccs_arr.shape}")
    logger.info(f"  RDKit FP    : {rdkit_arr.shape}")

    return desc_df, morgan_arr, maccs_arr, rdkit_arr, valid_mask
