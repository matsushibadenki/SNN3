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
        
        def hook(module, input, output):
            nonlocal total_ops, total_synapses
            
            if isinstance(output, tuple):
                spikes = output[0]
            else:
                spikes = output
            
            # スパイクが発生した場所でのみ演算
            active_neurons = spikes.sum().item()
            if hasattr(module, 'features'):
                total_ops += active_neurons * module.features
            total_synapses += spikes.numel()
        
        # フックを登録
        handles = []
        for module in model.modules():
            if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                handles.append(module.register_forward_hook(hook))
        
        # 推論実行
        with torch.no_grad():
            model(input_batch)
        
        # フックを削除
        for h in handles:
            h.remove()
        
        sparsity = 1.0 - (total_ops / max(total_synapses, 1))
        
        return {
            'total_ops': total_ops,
            'active_synapses': total_ops,
            'sparsity': sparsity,
            'total_synapses': total_synapses
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
        ann_ops = ann_params * batch_size
        
        energy_ratio = (snn_ops * 0.1) / (ann_ops * 1.0)
        efficiency_gain = (1.0 - energy_ratio) * 100
        
        return {
            'ann_ops': ann_ops,
            'energy_ratio': energy_ratio,
            'efficiency_gain': efficiency_gain
        }
