# snn_research/bio_models/simple_network.py
# Title: BioSNN (生物学的SNN)
# Description: 生物学的学習則を組み込んだシンプルな2層SNN。
# 変更点:
# - 強化学習ループに対応するため、推論(forward)と学習(update_weights)を分離。
# - forwardメソッドが中間層のスパイクも返すように変更。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .lif_neuron import BioLIFNeuron
from snn_research.learning_rules.base_rule import BioLearningRule

class BioSNN(nn.Module):
    """生物学的学習則で学習するシンプルなSNNモデル。"""
    def __init__(self, n_input: int, n_hidden: int, n_output: int, neuron_params: dict, learning_rule: BioLearningRule):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rule = learning_rule
        
        self.hidden_layer = BioLIFNeuron(n_hidden, neuron_params)
        self.output_layer = BioLIFNeuron(n_output, neuron_params)
        
        # 重み
        self.w1 = nn.Parameter(torch.rand(n_hidden, n_input) * 0.5)
        self.w2 = nn.Parameter(torch.rand(n_output, n_hidden) * 0.5)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """推論のみを実行し、出力スパイクと中間層のスパイクを返す。"""
        # 入力層 -> 隠れ層
        hidden_current = torch.matmul(self.w1, input_spikes)
        hidden_spikes = self.hidden_layer(hidden_current)
        
        # 出力層
        output_current = torch.matmul(self.w2, hidden_spikes)
        output_spikes = self.output_layer(output_current)
            
        return output_spikes, hidden_spikes
        
    def update_weights(
        self,
        pre_spikes_l1: torch.Tensor,
        post_spikes_l1: torch.Tensor,
        pre_spikes_l2: torch.Tensor,
        post_spikes_l2: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ):
        """学習則に基づいて重みを更新する。"""
        if not self.training:
            return

        # 隠れ層の重み更新
        dw1 = self.learning_rule.update(
            pre_spikes=pre_spikes_l1, 
            post_spikes=post_spikes_l1,
            weights=self.w1,
            optional_params=optional_params
        )
        # nn.Parameterの更新は .data を使う
        self.w1.data += dw1
        
        # 出力層の重み更新
        dw2 = self.learning_rule.update(
            pre_spikes=pre_spikes_l2,
            post_spikes=post_spikes_l2,
            weights=self.w2,
            optional_params=optional_params
        )
        self.w2.data += dw2

        # 重みが負にならないようにクリッピング
        self.w1.data.clamp_(min=0)
        self.w2.data.clamp_(min=0)
