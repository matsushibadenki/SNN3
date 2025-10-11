# ファイルパス: snn_research/learning_rules/causal_trace.py
# (更新)
# Title: 因果追跡クレジット割り当て (Causal Trace Credit Assignment)
# Description: 微分から脱却し、スパイクの因果連鎖を辿ってクレジットを割り当てる
#              新しい生物学的学習則。
# 改善点: 「適応的因果スパース化」のため、シナプスの因果的貢献度を追跡する機能を追加。
# 改善点 (v2): 階層的因果学習を実現するため、後段の層から前段の層へ
#              クレジット信号を逆伝播させる機能を追加。

import torch
from typing import Dict, Any, Optional, Tuple
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignment(RewardModulatedSTDP):
    """
    因果追跡の思想に基づき、適格性トレースを用いてクレジット（報酬）を割り当てる学習ルール。
    RewardModulatedSTDPを継承し、その責務をより明確にする。
    """
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, tau_eligibility: float, dt: float = 1.0):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, tau_eligibility, dt)
        self.causal_contribution: Optional[torch.Tensor] = None
        print("🧠 Causal Trace Credit Assignment learning rule initialized.")

    def _initialize_contribution_trace(self, weight_shape: tuple, device: torch.device):
        """因果的貢献度を記録するトレースを初期化する。"""
        self.causal_contribution = torch.zeros(weight_shape, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        報酬信号と後段からのクレジットに基づき重み変化量を計算し、
        前段へのクレジット信号を生成して返す。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (重み変化量, 前段へ伝えるクレジット信号)
        """
        # RewardModulatedSTDPの更新ロジックを呼び出し、基本的な重み変化量を取得
        dw = super().update(pre_spikes, post_spikes, weights, optional_params)

        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self._initialize_contribution_trace(weights.shape, weights.device)
        
        assert self.causal_contribution is not None, "Causal contribution trace not initialized."

        # 報酬があった場合、その更新の大きさを貢献度として記録
        if optional_params and optional_params.get("reward", 0.0) != 0.0:
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 階層的クレジット割り当てのための逆方向信号を計算
        # post-synapticニューロンの活動（または誤差）を、結合重みを介して
        # pre-synapticニューロンの「責任」として逆投影する。
        # ここでは、適格性トレースを誤差信号として利用する。
        if self.eligibility_trace is not None:
            # eligibility_trace: [post, pre], weights: [post, pre]
            # backward_credit: [pre]
            backward_credit = torch.einsum('ij,ij->j', self.eligibility_trace, weights)
        else:
            backward_credit = torch.zeros_like(pre_spikes)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        return dw, backward_credit
