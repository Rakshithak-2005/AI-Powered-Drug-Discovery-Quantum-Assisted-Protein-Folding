import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
    }
filepath="SMILES_Big_Data_Set.csv"
def load_and_featurize(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['SMILES', 'pIC50'])
    df['features'] = df['SMILES'].apply(smiles_to_features)
    df = df.dropna(subset=['features'])
    features_df = pd.json_normalize(df['features'])

    final_df = pd.concat([
        df[['SMILES', 'pIC50', 'num_atoms', 'logP']].reset_index(drop=True),
        features_df
    ], axis=1)

    return final_df
