# matsushibadenki/snn3/snn_research/benchmark/metrics.py
# ベンチマーク評価用のメトリクス関数

from typing import List, Any
import torch

def calculate_accuracy(true_labels: List[int], pred_labels: List[int]) -> float:
    """分類タスクの正解率を計算する。"""
    if len(true_labels) == 0:
        return 0.0
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    return correct / len(true_labels)

def calculate_perplexity(model: torch.nn.Module, dataloader: Any, device: str) -> float:
    """パープレキシティを計算するダミー関数。"""
    print("Warning: calculate_perplexity is a dummy function.")
    return 0.0

def calculate_energy_consumption(avg_spikes_per_sample: float) -> float:
    """エネルギー消費を計算するダミー関数。"""
    print("Warning: calculate_energy_consumption is a dummy function.")
    return 0.0
