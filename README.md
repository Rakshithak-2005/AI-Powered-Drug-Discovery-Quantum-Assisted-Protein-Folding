# 🧬 AI-Powered Drug Discovery & Quantum-Assisted Protein Folding

An end-to-end computational drug discovery and biophysics platform integrating **Classical Machine Learning**, **Deep Neural Networks**, **Variational Quantum Circuits (PennyLane)**, **SHAP Explainability**, and **Quantum-Assisted Protein Folding (VQE)** on an interactive **Streamlit** dashboard.

---

## 🌟 Highlights & Key Innovations

- **Zero Dummy Pipelines**: All models—including the Variational Quantum Classifier (VQC) and Hybrid Quantum Neural Network (QNN)—are **fully trained** using PyTorch automatic differentiation backpropagation.
- **Bemis-Murcko Scaffold Splitting**: Replaced naive random train/test splits with Murcko scaffold clustering to ensure held-out test sets evaluate true generalization to *unseen chemotypes*, eliminating molecular data leakage.
- **Explainable AI (SHAP)**: Integrated TreeSHAP to quantify global feature importance (TPSA, MolWt, logP, H-donors/acceptors) and per-molecule waterfall force plots explaining why individual drug candidates are predicted active or inactive.
- **Quantum-Assisted Protein Folding**: Implemented the Hydrophobic-Polar (HP) 2D lattice folding model with both exhaustive classical conformational search and a **Variational Quantum Eigensolver (VQE)** that simulates ground-state folding Hamiltonians.
- **Rigorous & Honest Scientific Benchmarking**: Evaluates all four models on identical held-out test splits with standard regression metrics ($R^2$, MAE, RMSE, MSE) and provides a scientifically honest comparative analysis of NISQ quantum limits vs classical baselines.

---

## 🚀 System Architecture & Capabilities

```
AI-Powered Drug Discovery & Quantum Protein Folding
├── 1. Data Ingestion & Cheminformatics (data_loader.py)
│    ├── 16,000+ compounds parsed & validated via RDKit
│    ├── Physicochemical descriptors: MolWt, logP, TPSA, H-Donors, H-Acceptors, Heavy Atoms
│    ├── Bemis-Murcko Scaffold Splitting (Gold standard out-of-distribution evaluation)
│    └── 2D Molecular Structure Rendering
│
├── 2. Predictive Bioactivity Modeling
│    ├── Random Forest Regressor (ml_model.py)
│    ├── Deep Neural Network / MLP (dl_model.py — TensorFlow/Keras)
│    ├── Variational Quantum Classifier (qml_model.py — PennyLane AngleEmbedding + StronglyEntanglingLayers)
│    └── Hybrid Quantum Neural Network (qnn_model.py — PennyLane TorchLayer + PyTorch)
│
├── 3. Explainability & Mechanistic Insights (ml_model.py & app.py)
│    ├── TreeSHAP feature attribution (global summary)
│    ├── Per-molecule waterfall plots (local feature attribution)
│    └── Structure-Activity Relationship (SAR) Pearson correlation plots
│
├── 4. Biophysical Protein Folding (protein_folding.py)
│    ├── Amino acid sequence to Hydrophobic-Polar (HP) lattice mapping
│    ├── Classical exhaustive conformational ground-state search
│    ├── Variational Quantum Eigensolver (VQE) optimization trajectory
│    └── 2D Lattice fold visualization with non-covalent contact energetics
│
└── 5. Interactive Web Application (app.py)
     └── 5 specialized tabs: Bioactivity, Protein Folding, SHAP XAI, Scientific Comparison, Methodology
```

---

## ⚙️ Quick Start & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, macOS, or Linux

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Rakshitha-11/AI-Powered-Drug-Discovery-Quantum-Assisted-Protein-Folding.git
cd AI-Powered-Drug-Discovery-Quantum-Assisted-Protein-Folding
pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Automated Test Suites
Run the comprehensive integration test covering all data splits, training loops, SHAP, and quantum folding:
```bash
python test_integration.py
```
Run the unit test suite verifying backward-compatibility and edge cases:
```bash
python run_full_tests.py
```

---

## 🧪 Detailed Module Overview

### 1. Data Loader & Scaffold Splitting (`data_loader.py`)
- **Dataset**: Tested on `SMILES_Big_Data_Set.csv` containing **16,088 compounds** with experimentally measured $pIC_{50}$ values.
- **Descriptor Calculation**: Automatically computes MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, and heavy atom counts using RDKit.
- **Scaffold Splitting**: Groups molecules by their core Bemis-Murcko rings and linkers. Molecules with the same scaffold never cross between train and test splits, preventing model overoptimism caused by random splitting.

### 2. Machine Learning & Explainability (`ml_model.py`)
- **Random Forest Regressor**: 100 estimators trained on scaled physicochemical descriptors.
- **SHAP Integration**: Computes exact Shapley values using TreeSHAP to identify which physical properties drive bioactivity predictions (e.g. TPSA and logP driving membrane permeability and target binding).

### 3. Deep Learning (`dl_model.py`)
- **Architecture**: 3-layer Multi-Layer Perceptron (128 -> 64 -> 32 -> 1) with Dropout (0.2) and Batch Normalization built in Keras/TensorFlow.
- **Loss**: Mean Squared Error minimized via Adam optimizer with Early Stopping.

### 4. Trained Variational Quantum Classifier (`qml_model.py`)
- **Quantum Circuit**: 4-qubit circuit parameterized with `AngleEmbedding` feature encoding and `StronglyEntanglingLayers` entangling ansatz.
- **Training**: Optimized using PyTorch's Adam optimizer over multiple epochs using exact backpropagation gradients through PennyLane's PyTorch interface. Returns training loss curves.

### 5. Trained Hybrid Quantum Neural Network (`qnn_model.py`)
- **Architecture**: `Linear(in_features -> 4) -> Tanh * pi -> TorchLayer(4 qubits) -> Linear(4 -> 1)`.
- **Training**: End-to-end backpropagation through both classical linear projections and quantum circuit parameters using PyTorch.

### 6. Quantum-Assisted Protein Folding (`protein_folding.py`)
- **HP Model**: Translates amino acid sequences (e.g. `MLPFAGV`) into hydrophobic ($H$) and polar ($P$) chains. Non-covalent topological $H-H$ contacts release $-1.0\,\text{unit}$ of energy.
- **Classical Solver**: Exhaustive self-avoiding walk search over 2D lattice coordinate directions ($U, D, L, R$).
- **VQE Solver**: Parameterized quantum rotation circuit evaluated against the contact Hamiltonian expectation value, iteratively optimized to discover low-energy native conformations.

---

## 📊 Scientific Disclosure & Quantum Realism

In accordance with rigorous computational chemistry standards:
1. **Classical vs Quantum Benchmarks**: Classical Random Forest and Deep Learning models achieve higher $R^2$ scores on tabular molecular descriptors than current 4-qubit simulated quantum circuits. This is expected given the low qubit count and absence of native quantum topological encoding for flat tabular features.
2. **Value of NISQ Implementations**: The quantum circuits implemented here serve as functional, trainable variational prototypes demonstrating how parameterized quantum circuits (PQCs) can interface with cheminformatics pipelines, paving the way for fault-tolerant quantum advantage when scaled to quantum hardware.

---

## 👥 Contributors

- **Rakshitha K**
- **Gagana K**
- **Shriya D Pai**
- **Vathsalya R**

**📬 Contact:**  
rakshithakrishnamurthy2005@gmail.com | gaganakp1609@gmail.com | shriyapai13@gmail.com | vathsalyar410@gmail.com

---

## 🪪 License
Distributed under the MIT License. See `LICENSE` for details.
