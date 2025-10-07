# matsushibadenki/snn3/snn_research/core/neurons.py
"""
AdaptiveLIFNeuron implementation based on expert feedback.
- device/dtype-aware
- vectorized updates (batch x units)
- docstrings and type hints
- BPTT-enabled state updates
- Correct surrogate gradient usage
"""
from typing import Optional, Tuple, Dict

import torch
from torch import Tensor, nn
from spikingjelly.activation_based import surrogate

class AdaptiveLIFNeuron(base.MemoryModule):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron with threshold adaptation.
    Designed for vectorized operations and to be BPTT-friendly.

    Args:
        features (int): Number of neurons in the layer.
        tau_mem (float): Membrane time constant.
        tau_adapt (float): Adaptation time constant.
        base_threshold (float): Initial and base value for the firing threshold.
        adaptation_strength (float): How much a spike increases the threshold.
        target_spike_rate (float): Target spike rate for homeostatic regulation.
    """
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        tau_adapt: float = 200.0,
        base_threshold: float = 1.0,
        adaptation_strength: float = 0.1,
        target_spike_rate: float = 0.02,
    ):
        super().__init__()
        self.features = features
        
        # Time constants and decay rates
        self.mem_decay = math.exp(-1.0 / tau_mem)
        self.adapt_decay = math.exp(-1.0 / tau_adapt)

        # Base threshold can be a learnable parameter
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        
        # Surrogate gradient function
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        # State variables (buffers)
        self.register_buffer("mem", torch.zeros(features))
        self.register_buffer("adaptive_threshold", torch.zeros(features))
        self.register_buffer("spikes", torch.zeros(features))

    def reset(self):
        """Resets the neuron's state variables."""
        super().reset()
        self.mem.zero_()
        self.adaptive_threshold.zero_()
        self.spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes one timestep of input current.

        Args:
            x (Tensor): Input current of shape (batch_size, features).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Spikes (Tensor): Output spikes of shape (batch_size, features).
                - Membrane potential (Tensor): Membrane potential of shape (batch_size, features).
        """
        # Ensure state buffers are correctly sized for the batch
        if self.mem.shape[0] != x.shape[0]:
            self.mem = torch.zeros_like(x)
            self.adaptive_threshold = torch.zeros_like(x)

        # Update membrane potential (BPTT-friendly)
        self.mem = self.mem * self.mem_decay + x
        
        # Calculate current total threshold
        current_threshold = self.base_threshold + self.adaptive_threshold

        # Generate spikes
        spike = self.surrogate_function(self.mem - current_threshold)
        
        # Store spikes for statistics (shape is unified)
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike

        # Reset membrane potential for spiking neurons (non-differentiable part)
        # We use detach() here because reset is a non-gradient event.
        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask)

        # Update adaptive threshold
        self.adaptive_threshold = self.adaptive_threshold * self.adapt_decay + self.adaptation_strength * spike
        
        return spike, self.mem