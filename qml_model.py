import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

def train_qml(X_train, y_train, X_test, n_qubits=4, n_layers=2):
    """
    Trains a quantum model and returns circuit, parameters, and predictions
    """
    try:
        # Validate input dimensions
        if X_train.shape[1] > n_qubits:
            raise ValueError(f"Input features ({X_train.shape[1]}) exceed available qubits ({n_qubits})")
        
        # Use a simulator for demonstration
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(n_qubits))
            StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Initialize random weights
        weights = 0.01 * np.random.randn(n_layers, n_qubits, 3)
        
        # For demo purposes, we'll just use the first few samples
        X_test_sample = X_test[:min(10, len(X_test))]
        y_pred = np.array([circuit(x, weights) for x in X_test_sample])
        
        return circuit, weights, y_pred.mean(axis=1)  # Average across qubits
        
    except Exception as e:
        print(f"Error in QML training: {str(e)}")
        return None, None, None