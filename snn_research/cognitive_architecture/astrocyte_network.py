# /snn_research/cognitive_architecture/astrocyte_network.py
#
# Phase 4: ニューロン群の活動を長期的に調整するアストロサイト・ネットワーク
#
# 機能:
# - グローバルな発火活動を監視し、ネットワーク全体の恒常性を維持する。
# - 特定のニューロン群の過活動や非活動を検知し、パラメータ（例: 発火閾値）を調整する。
# - 学習の安定化と、エネルギー効率の最適化に貢献する。
#
# 改善点:
# - ROADMAPの「Astrocyteによる動的ニューロン進化」に基づき、
#   活動が低いニューロン層をより表現力の高いモデル(Izhikevich)に
#   動的に置き換える自己進化機能を実装。
#
# 修正点:
# - mypyエラーを解消するため、_find_monitored_neurons内のリストに明示的な型アノテーションを追加。

import torch
import torch.nn as nn
from typing import List, Dict, Type

from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron

class AstrocyteNetwork:
    """
    SNN全体の活動を監視し、恒常性を維持するためのグリア細胞様ネットワーク。
    ニューロンモデルの動的進化機能も持つ。
    """
    def __init__(self, snn_model: nn.Module, monitoring_interval: int = 100, evolution_threshold: float = 0.1):
        self.snn_model = snn_model
        self.monitoring_interval = monitoring_interval
        self.evolution_threshold = evolution_threshold
        self.step_counter = 0
        
        # 監視対象となる適応的ニューロン層を登録
        self.monitored_neurons: List[nn.Module] = self._find_monitored_neurons()
        
        # 各層の長期的な平均発火率を記録する
        self.long_term_spike_rates: Dict[str, torch.Tensor] = {}
        print(f"✨ アストロサイト・ネットワークが {len(self.monitored_neurons)} 個のニューロン層の監視を開始しました。")

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _find_monitored_neurons(self) -> List[nn.Module]:
        """モデル内の監視対象ニューロン(LIF or Izhikevich)を再帰的に探索する。"""
        neurons: List[nn.Module] = []
        for module in self.snn_model.modules():
            if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                neurons.append(module)
        return neurons
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def step(self):
        """
        学習または推論の各ステップで呼び出され、内部カウンターをインクリメントする。
        一定間隔で監視・調整ロジックをトリガーする。
        """
        self.step_counter += 1
        if self.step_counter % self.monitoring_interval == 0:
            self.monitor_and_regulate()

    @torch.no_grad()
    def monitor_and_regulate(self):
        """
        ネットワーク全体の活動を監視し、必要に応じて調整・進化を行う。
        """
        print(f"\n🔬 アストロサイトによるグローバル活動監視 (ステップ: {self.step_counter})")
        
        # 監視対象リストを動的に更新
        self.monitored_neurons = self._find_monitored_neurons()

        for i, layer in enumerate(self.monitored_neurons):
            layer_name = f"{type(layer).__name__}_{i}"
            
            # ニューロンに記録されている実際の平均スパイク活動を直接使用する
            current_rate = layer.spikes.mean().item()
            
            # 長期的な発火率を更新 (指数移動平均)
            if layer_name in self.long_term_spike_rates:
                self.long_term_spike_rates[layer_name] = (
                    0.99 * self.long_term_spike_rates[layer_name] + 0.01 * torch.tensor(current_rate)
                )
            else:
                self.long_term_spike_rates[layer_name] = torch.tensor(current_rate)

            long_term_rate = self.long_term_spike_rates[layer_name].item()
            
            # --- ホメオスタティック可塑性 ---
            if isinstance(layer, AdaptiveLIFNeuron):
                target_rate = layer.target_spike_rate
                print(f"  - 層 {layer_name}: 長期平均発火率={long_term_rate:.4f} (目標: {target_rate:.4f})")

                if abs(long_term_rate - target_rate) > target_rate * 0.5:
                    adjustment_factor = 1.05 if long_term_rate > target_rate else 0.95
                    new_strength = layer.adaptation_strength * adjustment_factor
                    print(f"    - 恒常性調整: 適応強度を変更します: {layer.adaptation_strength:.4f} -> {new_strength:.4f}")
                    layer.adaptation_strength = new_strength
                
                # --- 動的ニューロン進化 ---
                # LIFニューロンの活動が著しく低い場合、Izhikevichニューロンに進化させる
                if long_term_rate < (target_rate * self.evolution_threshold):
                    print(f"    - 🧬 進化トリガー: {layer_name} の活動が低いため、Izhikevichニューロンへの進化を試みます。")
                    self._evolve_neuron_model(layer_to_evolve=layer, target_class=IzhikevichNeuron)
            else:
                print(f"  - 層 {layer_name}: 長期平均発火率={long_term_rate:.4f} (進化済み)")

    def _evolve_neuron_model(self, layer_to_evolve: nn.Module, target_class: Type[nn.Module]):
        """
        指定されたニューロン層を、新しいクラスのインスタンスに置き換える。
        """
        for name, module in self.snn_model.named_modules():
            # 親モジュール内の、置き換え対象のニューロン層を見つける
            for child_name, child_module in module.named_children():
                if child_module is layer_to_evolve:
                    print(f"    - 発見: '{name}' 内の '{child_name}' を進化させます。")
                    # 新しいニューロンインスタンスを作成
                    # 元のニューロンの 'features' 属性を引き継ぐ
                    if hasattr(layer_to_evolve, 'features'):
                        features = layer_to_evolve.features
                        new_neuron = target_class(features=features)
                        
                        # 親モジュールの属性を新しいニューロンに置き換え
                        setattr(module, child_name, new_neuron)
                        print(f"    - ✅ 成功: '{child_name}' は {target_class.__name__} に進化しました。")
                        return
        print(f"    - ❌ 失敗: モデル内で進化対象のニューロン層を置き換えられませんでした。")
