# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: 強化学習エージェント
# Description: 生物学的学習則（報酬変調型STDP）を用いて、環境との相互作用から
#              自律的に学習するエージェント。
# 修正点:
# - 階層的因果学習に対応したBioSNNの新しいインターフェースに合わせて、
#   モデルの初期化と重み更新の呼び出し方を修正し、mypyエラーを解消。

import torch
from typing import Dict, Any, List

from snn_research.bio_models.simple_network import BioSNN
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP

class ReinforcementLearnerAgent:
    """
    BioSNNと報酬変調型STDPを用いて強化学習を行うエージェント。
    """
    def __init__(self, input_size: int, output_size: int, device: str):
        self.device = device
        
        # 生物学的学習則を定義
        learning_rule = RewardModulatedSTDP(
            learning_rate=0.01,
            a_plus=1.0, a_minus=1.0,
            tau_trace=20.0,
            tau_eligibility=100.0
        )
        
        # ネットワークの層構造を定義
        hidden_size = (input_size + output_size) // 2
        layer_sizes = [input_size, hidden_size, output_size]
        
        # BioSNNモデルを初期化 (新しいインターフェースを使用)
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            learning_rule=learning_rule
        ).to(device)

        # 学習のための状態を保持
        self.last_all_spikes: List[torch.Tensor] = []


    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        現在の状態（入力スパイク）から、モデルの推論によって行動（出力スパイク）を決定する。
        """
        self.model.eval() # 推論モード
        with torch.no_grad():
            # 状態を時間的にエンコード（ここでは簡略化）
            input_spikes = state
            
            # モデルのフォワードパスを実行して行動を決定
            output_spikes, hidden_spikes_history = self.model(input_spikes)

            # 次の学習ステップのために状態を保存 (入力スパイクと全隠れ層のスパイク)
            self.last_all_spikes = [input_spikes] + hidden_spikes_history

        return output_spikes

    def learn(self, reward: float):
        """
        受け取った報酬信号を用いて、直前の行動を評価し、モデルの重みを更新する。
        """
        if not self.last_all_spikes:
            return

        self.model.train() # 学習モード
        
        optional_params = {"reward": reward}
        
        # 保存しておいた全層のスパイク活動と報酬を使って重みを更新 (新しいインターフェースを使用)
        self.model.update_weights(
            all_layer_spikes=self.last_all_spikes,
            optional_params=optional_params
        )
        
        # 更新後に状態をクリア
        self.last_all_spikes = []
