import pandas as pd
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def smiles_to_features(smiles):
    """
    Computes physicochemical molecular descriptors for a SMILES string using RDKit.
    Returns a dictionary of descriptors or None if SMILES is invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return {
            'MolWt': float(Descriptors.MolWt(mol)),
            'NumHDonors': int(Descriptors.NumHDonors(mol)),
            'NumHAcceptors': int(Descriptors.NumHAcceptors(mol)),
            'TPSA': float(Descriptors.TPSA(mol)),
            'calc_logP': float(Descriptors.MolLogP(mol)),
            'calc_num_atoms': int(mol.GetNumHeavyAtoms()),
        }
    except Exception:
        return None


def smiles_to_image(smiles, size=(300, 300)):
    """
    Converts a SMILES string to a 2D PIL image.
    Returns None if the SMILES string cannot be parsed.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            return Draw.MolToImage(mol, size=size)
    except Exception:
        pass
    return None


def generate_scaffold(smiles, include_chirality=False):
    """
    Generates the Bemis-Murcko generic scaffold for a SMILES string.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles.strip(), includeChirality=include_chirality)
    except Exception:
        return ""


def scaffold_split(df, test_size=0.2, random_state=42):
    """
    Performs Bemis-Murcko scaffold splitting on a molecular DataFrame.
    Compounds sharing the same core scaffold are grouped together, ensuring
    held-out evaluation tests genuine generalization to novel chemotypes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'SMILES' column.
    test_size : float, default=0.2
        Fraction of compounds allocated to the test set.
    random_state : int, default=42

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaff = generate_scaffold(row['SMILES'])
        scaffolds[scaff].append(idx)

    # Sort scaffold clusters by descending count
    sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)

    # Shuffle scaffolds with identical sizes deterministically
    rng = np.random.RandomState(random_state)
    rng.shuffle(sorted_scaffolds)

    train_indices = []
    test_indices = []
    target_test_count = int(len(df) * test_size)

    for scaff, indices in sorted_scaffolds:
        if len(test_indices) + len(indices) <= target_test_count or not test_indices:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    # Safety check: guarantee both sets have at least 1 sample
    if not train_indices:
        split_mid = len(test_indices) // 2
        train_indices = test_indices[:split_mid]
        test_indices = test_indices[split_mid:]
    elif not test_indices:
        split_mid = len(train_indices) // 2
        test_indices = train_indices[split_mid:]
        train_indices = train_indices[:split_mid]

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def train_test_split_dataset(df, split_method='scaffold', test_size=0.2, random_state=42):
    """
    Splits dataset into train and test sets using either:
    - 'scaffold': Bemis-Murcko scaffold split (industry gold standard for novel chemotype generalization)
    - 'random': Standard shuffled random split
    """
    if split_method.lower() == 'scaffold':
        return scaffold_split(df, test_size=test_size, random_state=random_state)
    else:
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(shuffled) * (1 - test_size))
        train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
        test_df = shuffled.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df


def load_and_featurize(filepath, max_rows=None):
    """
    Loads a dataset containing molecular SMILES and bioactivity (pIC50),
    computes RDKit molecular features, and returns a sanitized DataFrame.

    Parameters
    ----------
    filepath : str, Path, or file-like object
        Path or buffer to the CSV file.
    max_rows : int, optional
        Maximum number of rows to load/process for fast iteration.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['SMILES', 'pIC50', 'num_atoms', 'logP', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    """
    df = pd.read_csv(filepath)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    # Normalize column names case-insensitively
    col_map = {}
    for col in df.columns:
        c_clean = col.strip().lower()
        if c_clean in ['smiles', 'canonical_smiles', 'molecule_smiles']:
            col_map[col] = 'SMILES'
        elif c_clean in ['pic50', 'p_ic50', 'activity']:
            col_map[col] = 'pIC50'
        elif c_clean in ['logp', 'log_p', 'lipophilicity']:
            col_map[col] = 'logP'
        elif c_clean in ['num_atoms', 'natoms', 'heavy_atoms', 'atom_count']:
            col_map[col] = 'num_atoms'

    df = df.rename(columns=col_map)

    # Validate required columns
    if 'SMILES' not in df.columns:
        raise ValueError("Missing required column 'SMILES' in the dataset.")
    if 'pIC50' not in df.columns:
        raise ValueError("Missing required target column 'pIC50' in the dataset.")

    # Convert pIC50 to numeric and drop NaN
    df['pIC50'] = pd.to_numeric(df['pIC50'], errors='coerce')
    df = df.dropna(subset=['SMILES', 'pIC50'])
    df['SMILES'] = df['SMILES'].astype(str)

    # Compute RDKit features
    features_list = [smiles_to_features(s) for s in df['SMILES']]
    valid_mask = [f is not None for f in features_list]

    df = df[valid_mask].reset_index(drop=True)
    valid_features = [f for f in features_list if f is not None]

    if not valid_features:
        raise ValueError("No valid molecules could be parsed from the provided SMILES column.")

    features_df = pd.DataFrame(valid_features)

    # Use existing logP and num_atoms if present and valid; otherwise use RDKit calculated values
    if 'logP' in df.columns:
        logp_series = pd.to_numeric(df['logP'], errors='coerce')
        final_logp = logp_series.fillna(features_df['calc_logP'])
    else:
        final_logp = features_df['calc_logP']

    if 'num_atoms' in df.columns:
        atoms_series = pd.to_numeric(df['num_atoms'], errors='coerce')
        final_atoms = atoms_series.fillna(features_df['calc_num_atoms'])
    else:
        final_atoms = features_df['calc_num_atoms']

    final_df = pd.DataFrame({
        'SMILES': df['SMILES'].values,
        'pIC50': df['pIC50'].astype(float).values,
        'num_atoms': final_atoms.astype(int).values,
        'logP': final_logp.astype(float).values,
        'MolWt': features_df['MolWt'].astype(float).values,
        'NumHDonors': features_df['NumHDonors'].astype(int).values,
        'NumHAcceptors': features_df['NumHAcceptors'].astype(int).values,
        'TPSA': features_df['TPSA'].astype(float).values,
    })

    return final_df
