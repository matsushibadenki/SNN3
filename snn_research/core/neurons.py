# matsushibadenki/snn3/snn_research/core/neurons.py
"""
AdaptiveLIFNeuron and IzhikevichNeuron implementations based on expert feedback and documentation.
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
from spikingjelly.activation_based import surrogate, base # type: ignore

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
        noise_intensity: float = 0.0,
    ):
        super().__init__()
        self.features = features
        
        # Time constants and decay rates
        self.mem_decay = math.exp(-1.0 / tau_mem)

        # Base threshold can be a learnable parameter
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.noise_intensity = noise_intensity
        
        # Surrogate gradient function
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        # 【根本修正】バッチサイズに依存しないように状態変数をNoneで初期化
        self.register_buffer("mem", None)
        self.register_buffer("adaptive_threshold", None)
        self.register_buffer("spikes", torch.zeros(features))
        
        # 【追加】総スパイク数を追跡 (指示4)
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False  # 状態を保持するかどうか (指示3)

    def set_stateful(self, stateful: bool):
        """時系列データの処理モードを設定"""
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        """Resets the neuron's state variables."""
        super().reset()
        self.mem = None
        self.adaptive_threshold = None
        self.spikes.zero_()
        self.total_spikes.zero_() # 追加 (指示4)

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
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        if self.adaptive_threshold is None or self.adaptive_threshold.shape != x.shape:
            self.adaptive_threshold = torch.zeros_like(x)

        # Update membrane potential (BPTT-friendly)
        self.mem = self.mem * self.mem_decay + x
        
        # 確率的アンサンブルのためのノイズ注入
        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity

        # Calculate current total threshold
        current_threshold = self.base_threshold + self.adaptive_threshold

        # Generate spikes
        spike = self.surrogate_function(self.mem - current_threshold)
        
        # Store spikes for statistics (shape is unified)
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        # 【追加】統計を累積（勾配不要） (指示4)
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        # Reset membrane potential for spiking neurons (non-differentiable part)
        # We use detach() here because reset is a non-gradient event.
        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask)

        # Update adaptive threshold (no grad needed for this homeostatic process)
        if self.training:
            # スパイクのみdetachして、閾値の勾配は保持
            self.adaptive_threshold = (
                self.adaptive_threshold * self.mem_decay + 
                self.adaptation_strength * spike.detach()
            )
        else:
            # 推論時は勾配不要
            with torch.no_grad():
                self.adaptive_threshold = (
                    self.adaptive_threshold * self.mem_decay + 
                    self.adaptation_strength * spike
                )
        
        return spike, self.mem

class IzhikevichNeuron(base.MemoryModule):
    """
    Izhikevich neuron model, capable of producing a wide variety of firing patterns.
    Designed for vectorized operations and to be BPTT-friendly.

    Args:
        features (int): Number of neurons in the layer.
        a (float): Time scale of the recovery variable `u`.
        b (float): Sensitivity of `u` to the subthreshold fluctuations of `v`.
        c (float): After-spike reset value of the membrane potential `v`.
        d (float): After-spike reset of the recovery variable `u`.
    """
    def __init__(
        self,
        features: int,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
    ):
        super().__init__()
        self.features = features
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        self.v_peak = 30.0
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("v", None) # Membrane potential
        self.register_buffer("u", None) # Recovery variable
        self.register_buffer("spikes", torch.zeros(features))
        # 【追加】総スパイク数を追跡 (指示4)
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def reset(self):
        super().reset()
        self.v = None
        self.u = None
        self.spikes.zero_()
        self.total_spikes.zero_() # 追加 (指示4)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes one timestep of input current with corrected Izhikevich dynamics.

        Args:
            x (Tensor): Input current of shape (batch_size, features).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Spikes (Tensor): Output spikes of shape (batch_size, features).
                - Membrane potential (Tensor): Membrane potential `v`.
        """
        # 状態変数の初期化
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, self.c)
        if self.u is None or self.u.shape != x.shape:
            self.u = torch.full_like(x, self.b * self.c)

        # 【修正1】動力学を先に計算（リセット前の状態で）
        dt = 0.5  # 安定性のため小さめのステップ
        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + x
        du = self.a * (self.b * self.v - self.u)
        
        # 【修正2】状態を更新
        self.v = self.v + dv * dt
        self.u = self.u + du * dt
        
        # 【修正3】スパイク判定とリセット（更新後の状態で）
        spike = self.surrogate_function(self.v - self.v_peak)
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        # 【追加】統計を累積（勾配不要） (指示4)
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        
        reset_mask = (self.v >= self.v_peak).detach()
        self.v = torch.where(reset_mask, torch.full_like(self.v, self.c), self.v)
        self.u = torch.where(reset_mask, self.u + self.d, self.u)
        
        # 【修正4】クランプは最後に適用（リセット後の値に対して）
        self.v = torch.clamp(self.v, min=-100.0, max=50.0)

        return spike, self.v
