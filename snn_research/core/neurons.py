# matsushibadenki/snn3/snn_research/core/neurons.py
"""
AdaptiveLIFNeuron implementation based on expert feedback.
- BPTT-enabled state updates
- Correct surrogate gradient usage
- Docstrings and type hints
- Vectorized updates (batch x units)
- device/dtype-aware
"""
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import math
from spikingjelly.activation_based import surrogate, base

class AdaptiveLIFNeuron(base.MemoryModule):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron with threshold adaptation.
    Designed for vectorized operations and to be BPTT-friendly.

    Args:
        features (int): Number of neurons in the layer.
        tau_mem (float): Membrane time constant.
        base_threshold (float): Initial and base value for the firing threshold.
        adaptation_strength (float): How much a spike increases the threshold.
        target_spike_rate (float): Target spike rate for homeostatic regulation.
    """
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        base_threshold: float = 1.0,
        adaptation_strength: float = 0.1,
        target_spike_rate: float = 0.02,
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        noise_intensity: float = 0.0,
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    ):
        super().__init__()
        self.features = features
        
        # Time constants and decay rates
        self.mem_decay = math.exp(-1.0 / tau_mem)

        # Base threshold can be a learnable parameter
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.noise_intensity = noise_intensity
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        # Surrogate gradient function
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 【根本修正】バッチサイズに依存しないように状態変数をNoneで初期化
        self.register_buffer("mem", None)
        self.register_buffer("adaptive_threshold", None)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.register_buffer("spikes", torch.zeros(features))

    def reset(self):
        """Resets the neuron's state variables."""
        super().reset()
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.mem = None
        self.adaptive_threshold = None
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
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
        # 【根本修正】状態変数がNoneの場合、入力テンソルの形状に合わせて初回のみ初期化
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        if self.adaptive_threshold is None:
            self.adaptive_threshold = torch.zeros_like(x)

        # Update membrane potential (BPTT-friendly)
        self.mem = self.mem * self.mem_decay + x
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 確率的アンサンブルのためのノイズ注入
        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

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

        # Update adaptive threshold (no grad needed for this homeostatic process)
        with torch.no_grad():
            if self.training:
                self.adaptive_threshold = self.adaptive_threshold * self.mem_decay + self.adaptation_strength * spike
        
        return spike, self.mem
