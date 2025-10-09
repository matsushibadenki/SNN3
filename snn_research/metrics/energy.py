# snn_research/metrics/energy.py
"""
Energy efficiency metrics for Spiking Neural Networks.
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from torch import Tensor
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron


class EnergyMetrics:
    """SNNのエネルギー効率を測定するメトリクス"""
    
    @staticmethod
    def compute_synaptic_operations(model: nn.Module, input_batch: Tensor) -> Dict[str, float]:
        """
        シナプス演算回数（SNN特有の効率指標）を計算。
        
        Args:
            model: 評価対象のSNNモデル
            input_batch: 入力データ（input_ids）
            
        Returns:
            Dict[str, float]: {
                'total_ops': 総演算回数,
                'active_synapses': アクティブなシナプス数,
                'sparsity': スパース性（0-1）
            }
        """
        total_ops = 0
        total_synapses = 0
        
        # スパイク統計をリセット
        if hasattr(model, 'reset_spike_stats'):
            model.reset_spike_stats()

        # 推論実行
        with torch.no_grad():
            model(input_batch)
        
        # 総スパイク数を取得
        if hasattr(model, 'get_total_spikes'):
            total_spikes = model.get_total_spikes()
        else:
            total_spikes = 0
            
        # 総シナプス数を計算
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 簡略化モデル: SynOp ≒ Total Spikes
        # より正確には、各スパイクが後続層で引き起こす加算演算の数
        total_ops = total_spikes 

        sparsity = 1.0 - (total_ops / max(total_params, 1))
        
        return {
            'total_ops': total_ops,
            'active_synapses': total_ops, # 簡略化のため同値
            'sparsity': sparsity,
            'total_synapses': total_params
        }
    
    @staticmethod
    def compare_with_ann(snn_ops: float, ann_params: int, batch_size: int = 1) -> Dict[str, float]:
        """
        通常のANNと比較したエネルギー効率を推定。
        
        Args:
            snn_ops: SNNのシナプス演算回数
            ann_params: ANNのパラメータ数
            batch_size: バッチサイズ
            
        Returns:
            Dict[str, float]: {
                'ann_ops': ANNの演算回数（MAC）,
                'energy_ratio': エネルギー比（SNN/ANN）,
                'efficiency_gain': 効率向上率（%）
            }
        """
        # ANNは全パラメータで乗算・加算（MAC: Multiply-Accumulate）
        ann_ops = ann_params * batch_size
        
        # エネルギー比（目安: ACC演算 ≈ 1pJ, スパイク演算 ≈ 0.1pJ）
        # MACはADDより高コストなため、ここでは仮に10倍のコストとする
        energy_ratio = (snn_ops * 0.1) / (ann_ops * 1.0)
        efficiency_gain = (1.0 - energy_ratio) * 100
        
        return {
            'ann_ops': ann_ops,
            'energy_ratio': energy_ratio,
            'efficiency_gain': efficiency_gain
        }
