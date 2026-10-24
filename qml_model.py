import numpy as np
import torch
import pennylane as qml
from sklearn.preprocessing import StandardScaler


def train_qml(X_train, y_train, X_test, n_qubits=4, n_layers=2, epochs=15,
              max_train_samples=40, max_eval_samples=20, lr=0.05, random_seed=42):
    """
    Trains a Variational Quantum Circuit (VQC) on molecular features using PennyLane
    and PyTorch automatic differentiation backpropagation.

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
    lr : float, default=0.05
    random_seed : int, default=42

    Returns
    -------
    circuit : callable QNode
    weights : np.ndarray
    y_pred : np.ndarray
    history : list of float (training loss per epoch)
    """
    try:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Scale inputs into [-pi, pi] for angle embedding
        scaler = StandardScaler()
        X_tr_scaled = np.clip(scaler.fit_transform(X_train) * (np.pi / 3.0), -np.pi, np.pi)
        X_te_scaled = np.clip(scaler.transform(X_test) * (np.pi / 3.0), -np.pi, np.pi)

        # Truncate samples for interactive simulation speed on CPU
        n_tr = min(max_train_samples, len(X_tr_scaled))
        n_te = min(max_eval_samples, len(X_te_scaled))

        X_tr_tensor = torch.tensor(X_tr_scaled[:n_tr, :n_qubits], dtype=torch.float32)
        y_tr_tensor = torch.tensor(np.asarray(y_train)[:n_tr], dtype=torch.float32)
        X_te_tensor = torch.tensor(X_te_scaled[:n_te, :n_qubits], dtype=torch.float32)

        # Setup PennyLane device
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def q_circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Trainable variational weights & linear readout head (scale and bias)
        weights = torch.nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))
        bias = torch.nn.Parameter(torch.tensor(float(np.mean(y_train))))
        scale = torch.nn.Parameter(torch.tensor(1.0))

        optimizer = torch.optim.Adam([weights, bias, scale], lr=lr)
        loss_fn = torch.nn.MSELoss()

        history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = []
            for x in X_tr_tensor:
                expvals = torch.stack(q_circuit(x, weights))
                pred = torch.mean(expvals) * scale + bias
                preds.append(pred)
            pred_tensor = torch.stack(preds)
            loss = loss_fn(pred_tensor, y_tr_tensor)
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))

        # Evaluation on test set
        with torch.no_grad():
            test_preds = []
            for x in X_te_tensor:
                expvals = torch.stack(q_circuit(x, weights))
                pred = torch.mean(expvals) * scale + bias
                test_preds.append(pred.item())

        y_pred = np.array(test_preds, dtype=float)
        weights_np = weights.detach().cpu().numpy()

        return q_circuit, weights_np, y_pred, history

    except Exception as e:
        print(f"Error in QML training: {str(e)}")
        return None, None, None, []
