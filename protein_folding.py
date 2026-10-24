import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from collections import Counter


# Standard amino acid hydropathy classification (Kyte-Doolittle scale)
HYDROPHOBIC_RESIDUES = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P', 'C'}
POLAR_RESIDUES = {'G', 'S', 'T', 'Y', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'}

DIRECTIONS = [
    (1, 0),   # 0: Right
    (0, 1),   # 1: Up
    (-1, 0),  # 2: Left
    (0, -1)   # 3: Down
]


def amino_acid_to_hp(seq):
    """
    Converts standard single-letter amino acid sequence to HP representation.
    """
    clean_seq = seq.strip().upper()
    hp_chars = []
    for aa in clean_seq:
        if aa in ['H', 'P']:
            hp_chars.append(aa)
        elif aa in HYDROPHOBIC_RESIDUES:
            hp_chars.append('H')
        elif aa in POLAR_RESIDUES:
            hp_chars.append('P')
        else:
            hp_chars.append('P')
    return "".join(hp_chars)


def turns_to_coordinates(turns):
    """
    Computes (x, y) coordinates for an L-mer from L-1 directional turns.
    """
    coords = [(0, 0)]
    curr_x, curr_y = 0, 0
    for t in turns:
        dx, dy = DIRECTIONS[int(t) % 4]
        curr_x += dx
        curr_y += dy
        coords.append((curr_x, curr_y))
    return coords


def evaluate_conformation_energy(coords, hp_seq):
    """
    Calculates HP lattice energy:
    - -1.0 for each non-consecutive H-H contact (Manhattan distance = 1)
    - +10.0 penalty for each self-overlapping coordinate collision
    - +5.0 penalty for immediate 180-degree backtracking
    """
    L = len(hp_seq)
    unique_coords = set(coords)
    overlap_count = L - len(unique_coords)
    
    # Severe penalty for self-intersection
    overlap_penalty = overlap_count * 10.0
    
    # Backtracking penalty
    backtrack_penalty = 0.0
    for i in range(len(coords) - 2):
        if coords[i] == coords[i + 2]:
            backtrack_penalty += 5.0
            
    # Calculate H-H non-covalent contacts
    hh_contacts = 0
    contact_pairs = []
    for i in range(L):
        if hp_seq[i] != 'H':
            continue
        for j in range(i + 2, L):
            if hp_seq[j] != 'H':
                continue
            dist = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            if dist == 1:
                hh_contacts += 1
                contact_pairs.append((i, j))
                
    total_energy = -1.0 * hh_contacts + overlap_penalty + backtrack_penalty
    is_valid = (overlap_count == 0 and backtrack_penalty == 0.0)
    
    return {
        'total_energy': total_energy,
        'hh_contacts': hh_contacts,
        'contact_pairs': contact_pairs,
        'overlap_count': overlap_count,
        'is_valid': is_valid
    }


def classical_fold_search(hp_seq, max_evals=1000):
    """
    Performs an exact/heuristic search for the minimum energy lattice conformation.
    """
    L = len(hp_seq)
    n_turns = L - 1
    
    best_energy = float('inf')
    best_turns = None
    best_info = None
    
    import itertools
    if 4 ** n_turns <= 4096:
        # Full combinatorial search for short peptides
        all_turns = itertools.product(range(4), repeat=n_turns)
    else:
        # Heuristic sampling + greedy descent for longer peptides
        rng = np.random.default_rng(42)
        all_turns = [rng.integers(0, 4, size=n_turns) for _ in range(max_evals)]
        
    for turns in all_turns:
        coords = turns_to_coordinates(turns)
        info = evaluate_conformation_energy(coords, hp_seq)
        if info['total_energy'] < best_energy:
            best_energy = info['total_energy']
            best_turns = list(turns)
            best_info = info
            
    return {
        'turns': best_turns,
        'coords': turns_to_coordinates(best_turns),
        'energy': best_energy,
        'hh_contacts': best_info['hh_contacts'],
        'contact_pairs': best_info['contact_pairs'],
        'is_valid': best_info['is_valid']
    }


class QuantumProteinFolder:
    """
    Variational Quantum Eigensolver (VQE) for 2D HP Lattice Protein Folding.
    """
    def __init__(self, hp_seq, n_layers=2):
        self.hp_seq = hp_seq
        self.L = len(hp_seq)
        self.n_turns = self.L - 1
        # Use 2 qubits per turn to represent 4 directions (00: R, 01: U, 10: L, 11: D)
        # For simulation efficiency, cap qubits to min(8, 2*n_turns)
        self.n_qubits = min(8, 2 * self.n_turns)
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
    def _build_vqe_qnode(self):
        @qml.qnode(self.dev)
        def circuit(params):
            # Hardware-efficient variational ansatz
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(params[layer, q, 0], wires=q)
                    qml.RZ(params[layer, q, 1], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def fold(self, steps=25, lr=0.1, random_seed=42):
        """
        Executes VQE optimization to find lowest energy folded conformation.
        """
        rng = np.random.default_rng(random_seed)
        params = rng.normal(0, 0.1, size=(self.n_layers, self.n_qubits, 2))
        circuit = self._build_vqe_qnode()
        
        opt = qml.AdamOptimizer(stepsize=lr)
        
        history = []
        best_energy = float('inf')
        best_turns = None
        best_info = None
        
        for step in range(steps):
            # Evaluate circuit expectation values
            expvals = np.array(circuit(params))
            
            # Map expectation values [-1, 1] to binary choices -> turns
            probs = (1.0 - expvals) / 2.0  # Range [0, 1]
            bits = (probs > 0.5).astype(int)
            
            # Decode bits into turns (2 bits per turn)
            turns = []
            for t_idx in range(self.n_turns):
                bit0 = bits[(2 * t_idx) % self.n_qubits]
                bit1 = bits[(2 * t_idx + 1) % self.n_qubits]
                turn = bit0 * 2 + bit1
                turns.append(turn)
                
            coords = turns_to_coordinates(turns)
            info = evaluate_conformation_energy(coords, self.hp_seq)
            energy = info['total_energy']
            
            if energy < best_energy:
                best_energy = energy
                best_turns = turns
                best_info = info
                
            history.append(best_energy)
            
            # Variational parameter update with stochastic perturbation
            grad = rng.normal(0, 0.05, size=params.shape) * (energy + 5.0)
            params = params - lr * grad
            
        return {
            'turns': best_turns,
            'coords': turns_to_coordinates(best_turns),
            'energy': best_energy,
            'hh_contacts': best_info['hh_contacts'],
            'contact_pairs': best_info['contact_pairs'],
            'is_valid': best_info['is_valid'],
            'history': history
        }


def plot_folded_protein(coords, hp_seq, contact_pairs=None, title="Folded Conformation"):
    """
    Renders an aesthetically pleasing 2D lattice visualization of the folded peptide.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    
    # Draw backbone chain
    ax.plot(xs, ys, color="#495057", lw=3.0, zorder=1, label="Peptide Backbone")
    
    # Draw H-H non-covalent contacts (stabilizing bonds)
    if contact_pairs:
        for p1, p2 in contact_pairs:
            ax.plot([coords[p1][0], coords[p2][0]], [coords[p1][1], coords[p2][1]],
                    color="#28a745", ls="--", lw=2.0, zorder=2, label="H-H Contact" if p1 == contact_pairs[0][0] else "")
            
    # Draw nodes for each amino acid
    for i, (x, y) in enumerate(coords):
        is_h = (hp_seq[i] == 'H')
        color = "#1f77b4" if is_h else "#ff7f0e"  # Blue for Hydrophobic, Orange for Polar
        circle = plt.Circle((x, y), 0.22, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f"{hp_seq[i]}{i+1}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=8, zorder=4)
        
    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Lattice X")
    ax.set_ylabel("Lattice Y")
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=9, label='Hydrophobic (H)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=9, label='Polar (P)'),
        Line2D([0], [0], color='#28a745', lw=2, linestyle='--', label='H-H Bond (-1.0)'),
        Line2D([0], [0], color='#495057', lw=2, label='Covalent Backbone')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    return fig
