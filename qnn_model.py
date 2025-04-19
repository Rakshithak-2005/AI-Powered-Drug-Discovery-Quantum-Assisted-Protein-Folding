import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.qnn import TorchLayer
import torch
import torch.nn as nn

def create_qnn(X_train, y_train, X_test, n_qubits=4, n_layers=2):
    """
    Creates and trains a hybrid quantum-classical neural network
    """
    try:
        # Validate input dimensions
        if X_train.shape[1] > n_qubits:
            raise ValueError(f"Input features ({X_train.shape[1]}) exceed available qubits ({n_qubits})")
        
        # Quantum circuit definition
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(n_qubits))
            StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Hybrid model definition
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        qlayer = TorchLayer(quantum_circuit, weight_shapes)
        
        # For demo purposes, we'll use a small subset
        X_test_sample = torch.tensor(X_test[:min(10, len(X_test))], dtype=torch.float32)
        y_pred = qlayer(X_test_sample).detach().numpy().mean(axis=1)  # Average across qubits
        
        return qlayer, y_pred
        
    except Exception as e:
        print(f"Error creating QNN: {str(e)}")
        return None, None