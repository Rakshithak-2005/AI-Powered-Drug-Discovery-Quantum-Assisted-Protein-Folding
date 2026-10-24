import sys
import os
import pandas as pd
import numpy as np
from PIL import Image

from data_loader import load_and_featurize, smiles_to_features
from ml_model import train_ml
from dl_model import train_dl
from qml_model import train_qml
from qnn_model import create_qnn
from sklearn.decomposition import PCA


def test_data_loader_standard():
    print("\n--- Test 1: Standard Dataset Loading ---")
    df = load_and_featurize('SMILES_Big_Data_Set.csv', max_rows=50)
    assert len(df) == 50, f"Expected 50 rows, got {len(df)}"
    expected_cols = ['SMILES', 'pIC50', 'num_atoms', 'logP', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c}"
    print(f"PASS: Standard loading (50 rows): MolWt mean = {df['MolWt'].mean():.2f}")


def test_data_loader_missing_columns():
    print("\n--- Test 2: Dataset with missing optional columns (only SMILES and pIC50) ---")
    minimal_df = pd.DataFrame({
        'smiles': ['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'],
        'pic50': [5.2, 6.1, 4.8]
    })
    minimal_df.to_csv('test_minimal.csv', index=False)
    df = load_and_featurize('test_minimal.csv')
    assert len(df) == 3
    assert 'num_atoms' in df.columns
    assert 'logP' in df.columns
    assert 'MolWt' in df.columns
    print(f"PASS: Minimal dataset auto-calculated logP: {df['logP'].tolist()} and num_atoms: {df['num_atoms'].tolist()}")
    if os.path.exists('test_minimal.csv'):
        os.remove('test_minimal.csv')


def test_data_loader_invalid_smiles():
    print("\n--- Test 3: Handling invalid SMILES gracefully ---")
    invalid_df = pd.DataFrame({
        'SMILES': ['CC(=O)Oc1ccccc1C(=O)O', 'NOT_A_VALID_SMILES', 'c1ccccc1'],
        'pIC50': [5.0, 4.5, 3.2]
    })
    invalid_df.to_csv('test_invalid.csv', index=False)
    df = load_and_featurize('test_invalid.csv')
    assert len(df) == 2, f"Expected 2 valid molecules, got {len(df)}"
    print(f"PASS: Invalid SMILES correctly dropped. Kept {len(df)} compounds.")
    if os.path.exists('test_invalid.csv'):
        os.remove('test_invalid.csv')


def test_all_models():
    print("\n--- Test 4: Model Execution (ML, DL, QML, QNN) ---")
    df = load_and_featurize('SMILES_Big_Data_Set.csv', max_rows=60)
    feature_cols = ['num_atoms', 'logP', 'MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    X = df[feature_cols]
    y = df['pIC50']

    X_train, X_test = X.iloc[:45], X.iloc[45:]
    y_train, y_test = y.iloc[:45], y.iloc[45:]

    # Classical ML (Random Forest)
    print("Testing Random Forest...")
    rf_m, rf_pred = train_ml(X_train, y_train, X_test)
    assert len(rf_pred) == len(X_test), "RF prediction length mismatch"
    print(f"PASS: Random Forest predictions shape = {rf_pred.shape}")

    # Deep Learning
    print("Testing Deep Learning (Keras)...")
    dl_m, dl_pred = train_dl(X_train, y_train, X_test, epochs=15)
    assert len(dl_pred) == len(X_test), "DL prediction length mismatch"
    print(f"PASS: Deep Learning predictions shape = {dl_pred.shape}")

    # PCA for Quantum Models
    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # QML
    print("Testing QML (PennyLane Variational Circuit)...")
    qml_res = train_qml(X_train_pca, y_train, X_test_pca, max_eval_samples=5, epochs=2)
    if len(qml_res) == 4:
        q_c, q_w, q_pred, q_hist = qml_res
    else:
        q_c, q_w, q_pred = qml_res[:3]
    assert q_pred is not None and len(q_pred) == 5, "QML prediction issue"
    print(f"PASS: QML predictions (trained VQC) = {q_pred}")

    # QNN
    print("Testing QNN (PennyLane TorchLayer)...")
    qnn_res = create_qnn(X_train_pca, y_train, X_test_pca, max_eval_samples=5, epochs=2)
    if len(qnn_res) == 3:
        qnn_l, qnn_pred, qnn_hist = qnn_res
    else:
        qnn_l, qnn_pred = qnn_res[:2]
    assert qnn_pred is not None and len(qnn_pred) == 5, "QNN prediction issue"
    print(f"PASS: QNN predictions (trained hybrid QNN) = {qnn_pred}")


def test_molecular_visualization():
    print("\n--- Test 5: Molecular Visualization (RDKit to PIL Image) ---")
    from data_loader import smiles_to_image
    aspirin = 'CC(=O)Oc1ccccc1C(=O)O'
    img = smiles_to_image(aspirin, size=(200, 200))
    assert img is not None, "Failed to render aspirin image"
    assert isinstance(img, Image.Image), "Rendered object is not a PIL Image"
    print(f"PASS: Molecular image rendered successfully with size {img.size}")


if __name__ == '__main__':
    print("==================================================")
    print("STARTING TEST SUITE")
    print("==================================================")
    test_data_loader_standard()
    test_data_loader_missing_columns()
    test_data_loader_invalid_smiles()
    test_all_models()
    test_molecular_visualization()
    print("\n==================================================")
    print("ALL 5 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
