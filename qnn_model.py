import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.qnn import TorchLayer
from sklearn.preprocessing import StandardScaler


class HybridQuantumNeuralNetwork(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network architecture:
    Linear(in_features -> n_qubits) -> Tanh * pi -> TorchLayer (VQC) -> Linear(n_qubits -> 1)
    """
    def __init__(self, in_features, n_qubits, qlayer, init_bias=5.0):
        super().__init__()
        self.pre_linear = nn.Linear(in_features, n_qubits)
        self.qlayer = qlayer
        self.post_linear = nn.Linear(n_qubits, 1)
        nn.init.constant_(self.post_linear.bias, init_bias)

    def forward(self, x):
        h = torch.tanh(self.pre_linear(x)) * np.pi
        q_out = self.qlayer(h)
        return self.post_linear(q_out).squeeze(-1)


def train_qnn(X_train, y_train, X_test, n_qubits=4, n_layers=2, epochs=15,
              max_train_samples=40, max_eval_samples=20, lr=0.03, random_seed=42):
    """
    Trains a hybrid Quantum Neural Network (QNN) using PennyLane TorchLayer
    and PyTorch optimization on molecular descriptors.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    X_test : array-like of shape (n_samples_test, n_features)
    n_qubits : int, default=4
    n_layers : int, default=2
    epochs : int, default=15
    max_train_samples : int, default=40
    max_eval_samples : int, default=20
    lr : float, default=0.03
    random_seed : int, default=42

    Returns
    -------
    model : HybridQuantumNeuralNetwork
    y_pred : np.ndarray
    history : list of float (training loss per epoch)
    """
    try:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        n_tr = min(max_train_samples, len(X_tr_scaled))
        n_te = min(max_eval_samples, len(X_te_scaled))

        in_dim = min(n_qubits, X_tr_scaled.shape[1])
        X_tr_tensor = torch.tensor(X_tr_scaled[:n_tr, :in_dim], dtype=torch.float32)
        y_tr_tensor = torch.tensor(np.asarray(y_train)[:n_tr], dtype=torch.float32)
        X_te_tensor = torch.tensor(X_te_scaled[:n_te, :in_dim], dtype=torch.float32)

        # Quantum device & circuit
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(n_qubits))
            StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        qlayer = TorchLayer(quantum_circuit, weight_shapes)

        mean_y = float(np.mean(y_train))
        model = HybridQuantumNeuralNetwork(in_dim, n_qubits, qlayer, init_bias=mean_y)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(X_tr_tensor)
            loss = loss_fn(preds, y_tr_tensor)
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))

        with torch.no_grad():
            test_preds = model(X_te_tensor).cpu().numpy()

        y_pred = np.asarray(test_preds, dtype=float)
        return model, y_pred, history

    except Exception as e:
        print(f"Error in QNN training: {str(e)}")
        return None, None, []


# Backward-compatibility alias
def create_qnn(X_train, y_train, X_test, n_qubits=4, n_layers=2, **kwargs):
    model, y_pred, history = train_qnn(
        X_train, y_train, X_test,
        n_qubits=n_qubits,
        n_layers=n_layers,
        **kwargs
    )
    return model, y_pred, history
