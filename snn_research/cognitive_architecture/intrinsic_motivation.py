# matsushibadenki/snn2/snn_research/cognitive_architecture/intrinsic_motivation.py
# Phase 6: 内発的動機付けシステム
#
# 機能:
# - 自由エネルギー原理に基づき、予測誤差の減少（＝世界の理解が進むこと）を
#   内在的な報酬としてモデル化する。
# - ドーパミンやセロトニンといった神経修飾物質の役割を抽象的に再現する。
# - エージェントが未知の情報を探求し、学習を続けるための「好奇心」の源となる。
# 変更点:
# - PhysicsEvaluatorと連携し、物理法則の一致度も報酬に組み込むように進化。
# - update_errorをupdate_motivationに改名し、複数の情報源から動機を計算する。

import torch
from collections import deque
from typing import Deque, Dict

from .physics_evaluator import PhysicsEvaluator

class IntrinsicMotivationSystem:
    """
    予測誤差の変化と物理法則の一致度を監視し、内的な報酬信号を生成するシステム。
    """
    def __init__(self, physics_evaluator: PhysicsEvaluator, window_size: int = 100):
        """
        Args:
            physics_evaluator (PhysicsEvaluator): 物理法則を評価するモジュール。
            window_size (int): 予測誤差の履歴を保持するウィンドウサイズ。
        """
        self.error_history: Deque[float] = deque(maxlen=window_size)
        self.window_size = window_size
        self.last_reward = 0.0
        self.physics_evaluator = physics_evaluator

    def update_motivation(
        self,
        current_error: float,
        mem_sequence: torch.Tensor,
        spikes: torch.Tensor,
        prediction_reward_weight: float = 1.0,
        physics_reward_weight: float = 0.5
    ) -> float:
        """
        新しい観測に基づき、総合的な内在的報酬を計算する。

        Args:
            current_error (float): 最新の予測誤差（損失）。
            mem_sequence (torch.Tensor): SNNの膜電位の時系列。
            spikes (torch.Tensor): SNNのスパイク活動。
            prediction_reward_weight (float): 予測報酬の重み。
            physics_reward_weight (float): 物理報酬の重み。

        Returns:
            float: 計算された総合的な内在的報酬。
        """
        # 1. 予測誤差の減少から生じる報酬（世界の理解が進んだ喜び）
        prediction_reward = 0.0
        if len(self.error_history) > 0:
            past_average_error = sum(self.error_history) / len(self.error_history)
            prediction_reward = past_average_error - current_error
        self.error_history.append(current_error)

        # 2. 物理法則の一致度から生じる報酬（内部モデルの美しさ・効率性）
        physics_rewards = self.physics_evaluator.evaluate_physical_consistency(mem_sequence, spikes)
        avg_physics_reward = sum(physics_rewards.values()) / len(physics_rewards)

        # 3. 総合的な報酬の計算
        total_reward = (prediction_reward_weight * prediction_reward) + \
                       (physics_reward_weight * avg_physics_reward)

        self.last_reward = total_reward

        print(f"🧠 内的状態: 予測報酬[{prediction_reward:+.3f}], "
              f"物理報酬(滑らかさ:{physics_rewards['smoothness_reward']:.2f}, "
              f"スパース性:{physics_rewards['sparsity_reward']:.2f}), "
              f"総合報酬[{total_reward:+.3f}]")

        return total_reward

    def should_explore(self, reward_threshold: float = 0.1) -> bool:
        """
        現在の報酬レベルに基づき、エージェントが新しい探求活動を
        開始すべきかどうかを判断する。

        Returns:
            bool: 報酬が低く（＝退屈しており）、新しい刺激を求めるべきならTrue。
        """
        is_bored = abs(self.last_reward) < reward_threshold
        if is_bored and len(self.error_history) == self.window_size:
            print("🥱 システムは安定しており、退屈しています。新しい探求を推奨します。")
        return is_bored
