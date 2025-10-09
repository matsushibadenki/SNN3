# snn_research/learning_rules/causal_trace.py
# (新規作成)
# Title: 因果追跡クレジット割り当て (Causal Trace Credit Assignment)
# Description: 微分から脱却し、スパイクの因果連鎖を辿ってクレジットを割り当てる
#              新しい生物学的学習則。

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
        print("🧠 Causal Trace Credit Assignment learning rule initialized.")

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
        # 親クラス(RewardModulatedSTDP)のロジックをそのまま利用する
        return super().update(pre_spikes, post_spikes, weights, optional_params)