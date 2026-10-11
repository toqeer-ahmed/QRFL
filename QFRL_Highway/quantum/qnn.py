import sys
import types
import torch
import torch.nn as nn

# Safely ensure qiskit_ibm_runtime does not block torchquantum import if not installed
try:
    import qiskit_ibm_runtime
except ImportError:
    sys.modules['qiskit_ibm_runtime'] = types.ModuleType('qiskit_ibm_runtime')

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:
    import mock_torchquantum as tq
    import mock_torchquantum.functional as tqf

class Op1QAllLayer(nn.Module):
    """
    Wrapper for 1-qubit parameterized operations on all wires.
    Works seamlessly with real TorchQuantum (tq.TrainableOpAll) and mock (tq.Op1QAllLayer).
    """
    def __init__(self, op, n_wires, has_params=True):
        super().__init__()
        self.n_wires = n_wires
        self.has_params = has_params
        if hasattr(tq, 'TrainableOpAll'):
            self.layer = tq.TrainableOpAll(n_gate=n_wires, op=op)
        elif hasattr(tq, 'Op1QAllLayer'):
            self.layer = tq.Op1QAllLayer(op=op, n_wires=n_wires, has_params=has_params)
        else:
            raise AttributeError("Neither TrainableOpAll nor Op1QAllLayer found in torchquantum")

    def forward(self, q_device):
        return self.layer(q_device)

    @property
    def params(self):
        # Convenience accessor for testing/backward compatibility
        if hasattr(self.layer, 'params'):
            return self.layer.params
        if hasattr(self.layer, 'gate_all') and len(self.layer.gate_all) > 0:
            return self.layer.gate_all[0].params
        return None

class QuantumActor(nn.Module):
    """
    Hybrid Quantum-Classical Actor for QFRL.
    
    Architecture:
    1. Classical compression: Linear(150, 4) -> (IMPLEMENTATION CHOICE)
    2. Quantum encoding: Angle encoding on 4 qubits -> (PAPER-SPECIFIED configuration)
    3. Quantum circuit: 12 layers of Rx, Ry, Rz, CNOT -> (PAPER-SPECIFIED configuration, IMPLEMENTATION CHOICE for exact ordering)
    4. Measurement: Pauli-Z on 4 qubits -> (IMPLEMENTATION CHOICE)
    5. Classical expansion: Linear(4, 5) -> Tanh -> (IMPLEMENTATION CHOICE)
    """
    def __init__(self, state_dim=150, action_dim=5, n_qubits=4, n_layers=12, n_shots=2048):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_shots = n_shots
        
        # IMPLEMENTATION CHOICE: Classical compression R^150 -> R^4
        self.state_compression = nn.Sequential(
            nn.Linear(self.state_dim, self.n_qubits),
            # Use tanh to keep angles in a reasonable range [-1, 1] before scaling
            nn.Tanh() 
        )
        
        # QNN Components
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        
        # IMPLEMENTATION CHOICE: Exact gate ordering per layer (Rx, Ry, Rz, CNOT)
        # The paper specifies the use of these gates and 12 layers, but not the specific sequence.
        self.rx_layers = nn.ModuleList([Op1QAllLayer(op=tq.RX, n_wires=self.n_qubits, has_params=True) for _ in range(self.n_layers)])
        self.ry_layers = nn.ModuleList([Op1QAllLayer(op=tq.RY, n_wires=self.n_qubits, has_params=True) for _ in range(self.n_layers)])
        self.rz_layers = nn.ModuleList([Op1QAllLayer(op=tq.RZ, n_wires=self.n_qubits, has_params=True) for _ in range(self.n_layers)])
        
        # IMPLEMENTATION CHOICE: Classical expansion R^4 -> R^5 -> Tanh
        self.action_expansion = nn.Sequential(
            nn.Linear(self.n_qubits, self.action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        batch_size = state.shape[0]
        
        # 1. Compress state
        # State angles will be bounded roughly in [-pi, pi] by scaling
        angles = self.state_compression(state) * torch.pi 
        
        # Reset quantum device
        self.q_device.reset_states(batch_size)
        
        # Save for mock gradient flow if using mock
        if hasattr(self.q_device, 'mock_angles'):
            self.q_device.mock_angles = angles
        
        # 2. PAPER-SPECIFIED: Hadamard initialization
        for w in range(self.n_qubits):
            tqf.hadamard(self.q_device, wires=w)
            
        # 3. PAPER-SPECIFIED: Angle encoding
        for w in range(self.n_qubits):
            tqf.ry(self.q_device, wires=w, params=angles[:, w])
            
        # 4. PAPER-SPECIFIED: 12 Layers with Rx, Ry, Rz and CNOT
        for i in range(self.n_layers):
            self.rx_layers[i](self.q_device)
            self.ry_layers[i](self.q_device)
            self.rz_layers[i](self.q_device)
            
            # Entanglement (CNOT ring)
            for w in range(self.n_qubits):
                tqf.cnot(self.q_device, wires=[w, (w + 1) % self.n_qubits])
                
        # 5. Measurement (IMPLEMENTATION CHOICE: 1 Pauli-Z per qubit)
        # Note: In TorchQuantum, forward pass calculates exact expectations algebraically. 
        # For training, it uses exact gradients. n_shots is mostly for real-device execution/sampling simulation.
        # We perform exact statevector simulation as supported by TorchQuantum.
        exp_vals = self.measure(self.q_device)
        
        # 6. Action Expansion (IMPLEMENTATION CHOICE)
        action = self.action_expansion(exp_vals)
        
        return action
