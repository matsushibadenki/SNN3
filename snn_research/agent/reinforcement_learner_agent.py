# matsushibadenki/snn3/snn_research/agent/reinforcement_learner_agent.py
# Title: 強化学習エージェント
# Description: 生物学的学習則（報酬変調型STDP）を用いて、環境との相互作用から
#              自律的に学習するエージェント。

import torch
from typing import Dict, Any

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
        
        # BioSNNモデルを初期化
        self.model = BioSNN(
            n_input=input_size,
            n_hidden= (input_size + output_size) // 2,
            n_output=output_size,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            learning_rule=learning_rule
        ).to(device)

        # 学習のための状態を保持
        self.last_input_spikes: torch.Tensor = torch.zeros(input_size, device=device)
        self.last_hidden_spikes: torch.Tensor = torch.zeros((input_size + output_size) // 2, device=device)
        self.last_output_spikes: torch.Tensor = torch.zeros(output_size, device=device)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        現在の状態（入力スパイク）から、モデルの推論によって行動（出力スパイク）を決定する。
        """
        self.model.eval() # 推論モード
        with torch.no_grad():
            # 状態を時間的にエンコード（ここでは簡略化）
            input_spikes = state
            
            # モデルのフォワードパスを実行して行動を決定
            output_spikes, hidden_spikes = self.model(input_spikes)

            # 次の学習ステップのために状態を保存
            self.last_input_spikes = input_spikes
            self.last_hidden_spikes = hidden_spikes
            self.last_output_spikes = output_spikes

        return output_spikes

    def learn(self, reward: float):
        """
        受け取った報酬信号を用いて、直前の行動を評価し、モデルの重みを更新する。
        """
        self.model.train() # 学習モード
        
        optional_params = {"reward": reward}
        
        # 保存しておいたスパイク活動と報酬を使って重みを更新
        self.model.update_weights(
            pre_spikes_l1=self.last_input_spikes,
            post_spikes_l1=self.last_hidden_spikes,
            pre_spikes_l2=self.last_hidden_spikes,
            post_spikes_l2=self.last_output_spikes,
            optional_params=optional_params
        )
