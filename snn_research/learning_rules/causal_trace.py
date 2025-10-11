# snn_research/learning_rules/causal_trace.py
# (新規作成)
# Title: 因果追跡クレジット割り当て (Causal Trace Credit Assignment)
# Description: 微分から脱却し、スパイクの因果連鎖を辿ってクレジットを割り当てる
#              新しい生物学的学習則。
# 改善点: 「適応的因果スパース化」のため、シナプスの因果的貢献度を追跡する機能を追加。

import torch
from typing import Dict, Any, Optional
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignment(RewardModulatedSTDP):
    """
    因果追跡の思想に基づき、適格性トレースを用いてクレジット（報酬）を割り当てる学習ルール。
    RewardModulatedSTDPを継承し、その責務をより明確にする。
    """
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, tau_eligibility: float, dt: float = 1.0):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, tau_eligibility, dt)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.causal_contribution: Optional[torch.Tensor] = None
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        print("🧠 Causal Trace Credit Assignment learning rule initialized.")

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _initialize_contribution_trace(self, weight_shape: tuple, device: torch.device):
        """因果的貢献度を記録するトレースを初期化する。"""
        self.causal_contribution = torch.zeros(weight_shape, device=device)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        報酬信号に基づいて重み変化量を計算する。
        このメソッドは、どのシナプスが最近の活動に「因果的」に関与したかを
        適格性トレースとして記録し、報酬が与えられた際にその貢献度に応じて重みを更新する。
        """
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        dw = super().update(pre_spikes, post_spikes, weights, optional_params)
        
        # 貢献度トレースの初期化
        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self._initialize_contribution_trace(weights.shape, weights.device)
        
        assert self.causal_contribution is not None, "Causal contribution trace not initialized."

        # 報酬があった場合、その更新の大きさを貢献度として記録（指数移動平均）
        if optional_params and optional_params.get("reward", 0.0) != 0.0:
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

        return dw
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
