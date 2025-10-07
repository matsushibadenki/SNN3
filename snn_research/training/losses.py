# matsushibadenki/snn3/snn_research/training/losses.py
# SNN学習で使用する損失関数
# 
# BugFix: DistillationLossでattention_maskを使い、パディングトークンを損失計算から除外する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import PreTrainedTokenizerBase

def _calculate_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """モデルの重みのL1ノルムを計算し、スパース性を促進する。"""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.tensor(0.0)

    device = params[0].device
    l1_norm = sum(
        (p.abs().sum() for p in params),
        start=torch.tensor(0.0, device=device)
    )
    return l1_norm

class CombinedLoss(nn.Module):
    """クロスエントロピー損失、各種正則化を組み合わせた損失関数。"""
    def __init__(self, ce_weight: float, spike_reg_weight: float, sparsity_reg_weight: float, mem_reg_weight: float, tokenizer: PreTrainedTokenizerBase, target_spike_rate: float = 0.02):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {'ce': ce_weight, 'spike_reg': spike_reg_weight, 'sparsity_reg': sparsity_reg_weight, 'mem_reg': mem_reg_weight}
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        sparsity_loss = _calculate_sparsity_loss(model)

        mem_reg_loss = torch.mean(mem**2)
        
        total_loss = (self.weights['ce'] * ce_loss + 
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss)
        
        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss, 'sparsity_loss': sparsity_loss,
            'mem_reg_loss': mem_reg_loss, 'spike_rate': spike_rate
        }

class DistillationLoss(nn.Module):
    """知識蒸留のための損失関数（各種正則化付き）。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float, distill_weight: float,
                 spike_reg_weight: float, sparsity_reg_weight: float, mem_reg_weight: float, temperature: float, target_spike_rate: float = 0.02):
        super().__init__()
        student_pad_id = tokenizer.pad_token_id
        self.temperature = temperature
        self.weights = {'ce': ce_weight, 'distill': distill_weight, 'spike_reg': spike_reg_weight, 'sparsity_reg': sparsity_reg_weight, 'mem_reg': mem_reg_weight}
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_pad_id if student_pad_id is not None else -100)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.distill_loss_fn = nn.KLDivLoss(reduction='none') # reductionを'none'に変更
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.target_spike_rate = target_spike_rate

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        assert student_logits.shape == teacher_logits.shape, \
            f"Shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}"
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # CE損失はignore_indexでパディングを自動的に無視する
        ce_loss = self.ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
        
        # Distillation損失でパディングを無視する
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # まず各要素の損失を計算
        distill_loss_unreduced = self.distill_loss_fn(soft_student_log_probs, soft_teacher_probs).sum(dim=-1)
        
        if attention_mask is None:
            # マスクがない場合は、ターゲットIDに基づいてマスクを作成
            mask = (targets != self.ce_loss_fn.ignore_index)
        else:
            mask = attention_mask.bool()
            
        # マスクを適用して、有効なトークンのみの損失を平均する
        masked_distill_loss = distill_loss_unreduced.where(mask, torch.tensor(0.0, device=distill_loss_unreduced.device))
        
        num_valid_tokens = mask.sum()
        if num_valid_tokens > 0:
            distill_loss = masked_distill_loss.sum() / num_valid_tokens
        else:
            distill_loss = torch.tensor(0.0, device=student_logits.device)
            
        distill_loss = distill_loss * (self.temperature ** 2)

        spike_rate = spikes.mean()
        target_spike_rate = torch.tensor(self.target_spike_rate, device=spikes.device)
        spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)

        sparsity_loss = _calculate_sparsity_loss(model)
        
        mem_reg_loss = torch.mean(mem**2)

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['distill'] * distill_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'distill_loss': distill_loss, 'spike_reg_loss': spike_reg_loss,
            'sparsity_loss': sparsity_loss, 'mem_reg_loss': mem_reg_loss
        }
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
class SelfSupervisedLoss(nn.Module):
    """
    時間的自己教師あり学習のための損失関数。
    """
    def __init__(self, prediction_weight: float, spike_reg_weight: float, sparsity_reg_weight: float, mem_reg_weight: float, tokenizer: PreTrainedTokenizerBase, target_spike_rate: float = 0.02):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.prediction_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'prediction': prediction_weight,
            'spike_reg': spike_reg_weight,
            'sparsity_reg': sparsity_reg_weight,
            'mem_reg': mem_reg_weight
        }
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module) -> dict:
        prediction_loss = self.prediction_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        sparsity_loss = _calculate_sparsity_loss(model)

        mem_reg_loss = torch.mean(mem**2)
        
        total_loss = (self.weights['prediction'] * prediction_loss + 
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss)
        
        return {
            'total': total_loss, 'prediction_loss': prediction_loss,
            'spike_reg_loss': spike_reg_loss, 'sparsity_loss': sparsity_loss,
            'mem_reg_loss': mem_reg_loss, 'spike_rate': spike_rate
        }


class PhysicsInformedLoss(nn.Module):
    """
    物理法則（膜電位の滑らかさ）を制約として組み込んだ損失関数。
    """
    def __init__(self, ce_weight: float, spike_reg_weight: float, mem_smoothness_weight: float, tokenizer: PreTrainedTokenizerBase, target_spike_rate: float = 0.02):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'ce': ce_weight,
            'spike_reg': spike_reg_weight,
            'mem_smoothness': mem_smoothness_weight,
        }
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem_sequence: torch.Tensor, model: nn.Module) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        if mem_sequence.numel() > 1:
            mem_diff = torch.diff(mem_sequence)
            mem_smoothness_loss = torch.mean(mem_diff**2)
        else:
            mem_smoothness_loss = torch.tensor(0.0, device=logits.device)

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['mem_smoothness'] * mem_smoothness_loss)
        
        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss,
            'mem_smoothness_loss': mem_smoothness_loss,
            'spike_rate': spike_rate
        }

class PlannerLoss(nn.Module):
    """
    プランナーSNNの学習用損失関数。
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, predicted_logits: torch.Tensor, target_plan: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = target_plan[:, 0]
        
        loss = self.loss_fn(predicted_logits, target)
        
        return {'total': loss, 'planner_loss': loss}
