# ファイルパス: snn_research/bio_models/simple_network.py
# タイトル: BioSNN (生物学的SNN)
# (省略...)
# 改善点:
# - ROADMAPフェーズ2「階層的因果学習」に基づき、複数層に対応できるように拡張。
# - update_weightsメソッドを一般化し、深いネットワークでの信用割り当てを可能にした。
# 改善点 (v2): 「適応的因果スパース化」を実装。貢献度の低いシナプスの学習を抑制する。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

from .lif_neuron import BioLIFNeuron
from snn_research.learning_rules.base_rule import BioLearningRule
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignment
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

class BioSNN(nn.Module):
    """生物学的学習則で学習する、複数層に対応したSNNモデル。"""
    def __init__(self, layer_sizes: List[int], neuron_params: dict, learning_rule: BioLearningRule, 
                 sparsification_config: Optional[Dict[str, Any]] = None): # ◾️ 追加
        super().__init__()
        self.layer_sizes = layer_sizes
        self.learning_rule = learning_rule
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.sparsification_enabled = sparsification_config.get("enabled", False) if sparsification_config else False
        self.contribution_threshold = sparsification_config.get("contribution_threshold", 0.0) if sparsification_config else 0.0
        if self.sparsification_enabled:
            print(f"🧬 適応的因果スパース化が有効です (貢献度閾値: {self.contribution_threshold})")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        # 層と重みのリストを作成
        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BioLIFNeuron(layer_sizes[i+1], neuron_params))
            # 重みをParameterとして登録
            weight = nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i]) * 0.5)
            self.weights.append(weight)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """推論のみを実行し、最終出力スパイクと各層のスパイク履歴を返す。"""
        hidden_spikes_history = []
        current_spikes = input_spikes
        
        for i, layer in enumerate(self.layers):
            current = torch.matmul(self.weights[i], current_spikes)
            current_spikes = layer(current)
            hidden_spikes_history.append(current_spikes)
            
        return current_spikes, hidden_spikes_history
        
    def update_weights(
        self,
        all_layer_spikes: List[torch.Tensor],
        optional_params: Optional[Dict[str, Any]] = None
    ):
        """学習則に基づいて全層の重みを更新する。"""
        if not self.training:
            return

        # 各層の重みを順番に更新
        for i in range(len(self.weights)):
            # 入力層のスパイクは all_layer_spikes の先頭に追加する
            pre_spikes = all_layer_spikes[i]
            post_spikes = all_layer_spikes[i+1]
            
            dw = self.learning_rule.update(
                pre_spikes=pre_spikes, 
                post_spikes=post_spikes,
                weights=self.weights[i],
                optional_params=optional_params
            )

            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            # 適応的因果スパース化のロジック
            if self.sparsification_enabled and isinstance(self.learning_rule, CausalTraceCreditAssignment):
                if self.learning_rule.causal_contribution is not None:
                    # 貢献度が閾値以下のシナプスの学習を抑制（ゲーティング）
                    contribution_mask = self.learning_rule.causal_contribution > self.contribution_threshold
                    dw = dw * contribution_mask
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

            # nn.Parameterの更新は .data を使う
            self.weights[i].data += dw
            # 重みが負にならないようにクリッピング
            self.weights[i].data.clamp_(min=0)
