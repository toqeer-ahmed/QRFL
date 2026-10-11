import torch
import torch.nn as nn

__version__ = "0.1.7-mock"

class QuantumDevice(nn.Module):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.batch_size = 1
        
    def reset_states(self, batch_size):
        self.batch_size = batch_size

class MeasureAll(nn.Module):
    def __init__(self, obs):
        super().__init__()
        self.obs = obs
        
    def forward(self, q_device):
        # Mock measurement: return something differentiable for testing
        # We can just return the sum of some parameters or states so gradients flow.
        # But we don't have access to the actual parameters easily in this mock,
        # so let's just make it return a tensor that requires grad if needed.
        # Actually, in qnn.py, we do:
        # angles = self.state_compression(state)
        # We can just return `angles` to keep the graph connected!
        # Let's add angles to q_device
        if hasattr(q_device, 'mock_angles'):
            return q_device.mock_angles
        return torch.rand((q_device.batch_size, q_device.n_wires), requires_grad=True) * 2 - 1

class Observable:
    pass

PauliZ = Observable()

class Op1QAllLayer(nn.Module):
    def __init__(self, n_gate=None, op=None, n_wires=None, has_params=True):
        super().__init__()
        self.op = op
        self.n_wires = n_wires if n_wires is not None else n_gate
        self.n_gate = self.n_wires
        self.has_params = has_params
        if has_params and self.n_wires is not None:
            self.params = nn.Parameter(torch.randn(self.n_wires))
            
    def forward(self, q_device):
        pass

# Operations constants
RX = "RX"
RY = "RY"
RZ = "RZ"

TrainableOpAll = Op1QAllLayer
