# matsushibadenki/snn3/snn_research/training/trainers.py
# SNNモデルの学習と評価ループを管理するTrainerクラス (モニタリング・評価機能完備)
# mypyエラー修正: 削除されていたPlannerTrainerを復元。
#                 MetaCognitiveSNNのメソッド呼び出しを修正。
#                 PlannerTrainer内の構文エラーを修正。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import os
import collections
from tqdm import tqdm  # type: ignore
from typing import Tuple, Dict, Any, Optional, cast
import shutil
import time
from torch.optim import Adam

from snn_research.training.losses import SpikeRateLoss
from snn_research.training.losses import SpikeRegularizationLoss
from snn_research.training.losses import CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from torch.utils.tensorboard import SummaryWriter




class BreakthroughTrainer:
    """モニタリングと評価機能を完備した、SNNの統合トレーニングシステム。"""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: str,
                 grad_clip_norm: float, rank: int, use_amp: bool, log_dir: str,
                 astrocyte_network: Optional[AstrocyteNetwork] = None,
                 meta_cognitive_snn: Optional[MetaCognitiveSNN] = None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank
        self.use_amp = use_amp and self.device != 'mps'
        self.astrocyte_network = astrocyte_network
        self.meta_cognitive_snn = meta_cognitive_snn
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_metric = float('inf')
        
        if self.rank in [-1, 0]:
            self.writer = SummaryWriter(log_dir)
            print(f"✅ TensorBoard logging enabled. Log directory: {log_dir}")

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        start_time = time.time()
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                logits, spikes, mem = self.model(input_ids, return_spikes=True)
                loss_dict = self.criterion(logits, target_ids, spikes, mem, self.model)
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            if self.astrocyte_network:
                self.astrocyte_network.step()
            if self.meta_cognitive_snn:
                end_time = time.time()
                computation_time = end_time - start_time
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                        ignore_idx = self.criterion.ce_loss_fn.ignore_index
                        mask = target_ids != ignore_idx
                        num_masked_elements = cast(torch.Tensor, mask).sum()
                        accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                        loss_dict['accuracy'] = accuracy.item()
                
                self.meta_cognitive_snn.update_metadata(
                    loss=loss_dict['total'].item(),
                    computation_time=computation_time,
                    accuracy=loss_dict.get('accuracy', 0.0)
                )

        with torch.no_grad():
            if 'accuracy' not in loss_dict:
                preds = torch.argmax(logits, dim=-1)
                if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                    ignore_idx = self.criterion.ce_loss_fn.ignore_index
                    mask = target_ids != ignore_idx
                    num_masked_elements = cast(torch.Tensor, mask).sum()
                    accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                    loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        self.model.train()
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value
            progress_bar.set_postfix({k: v / (progress_bar.n + 1) for k, v in total_metrics.items()})

        if self.scheduler: self.scheduler.step()
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0]:
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            self.writer.add_scalar('Train/learning_rate', self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_metrics

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        self.model.eval()
        with torch.no_grad():
            for batch in progress_bar:
                metrics = self._run_step(batch, is_train=False)
                for key, value in metrics.items(): total_metrics[key] += value
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0]:
            print(f"Epoch {epoch} Validation Results: " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
        
        return avg_metrics

    def save_checkpoint(self, path: str, epoch: int, metric_value: float, **kwargs: Any):
        if self.rank in [-1, 0]:
            model_to_save = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            buffer_names = {name for name, _ in model_to_save.named_buffers() if 'mem' not in name}
            model_state = {k: v for k, v in model_to_save.state_dict().items() if k not in buffer_names}

            state = {
                'epoch': epoch, 'model_state_dict': model_state, 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': self.best_metric
            }
            if self.use_amp: state['scaler_state_dict'] = self.scaler.state_dict()
            if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            print(f"✅ チェックポイントを '{path}' に保存しました (Epoch: {epoch})。")
            
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
                temp_state_for_best = {'model_state_dict': model_state, **kwargs}
                torch.save(temp_state_for_best, best_path)
                print(f"🏆 新しいベストモデルを '{best_path}' に保存しました (Metric: {metric_value:.4f})。")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            print(f"⚠️ チェックポイントファイルが見つかりません: {path}。最初から学習を開始します。")
            return 0
            
        checkpoint = torch.load(path, map_location=self.device)
        model_to_load = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_metric = checkpoint.get('best_metric', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ チェックポイント '{path}' を正常にロードしました。Epoch {start_epoch} から学習を再開します。")
        return start_epoch


class DistillationTrainer(BreakthroughTrainer):
    """知識蒸留に特化したトレーナー。"""
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, teacher_model: Optional[nn.Module] = None) -> Dict[str, float]:
        """知識蒸留のための学習ループ。"""
        final_metrics: Dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            final_metrics = self.evaluate(val_loader, epoch)
        return final_metrics

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        if is_train: self.model.train()
        else: self.model.eval()
            
        student_input, student_target, teacher_logits = [t.to(self.device) for t in batch]

        with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                student_logits, spikes, mem = self.model(student_input, return_spikes=True)
                
                assert isinstance(self.criterion, DistillationLoss)
                loss_dict = self.criterion(
                    student_logits=student_logits, teacher_logits=teacher_logits, targets=student_target,
                    spikes=spikes, mem=mem, model=self.model
                )
        
        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.astrocyte_network: self.astrocyte_network.step()
        
        return {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class SelfSupervisedTrainer(BreakthroughTrainer):
    """自己教師あり学習に特化したトレーナー。"""
    pass

class PhysicsInformedTrainer(BreakthroughTrainer):
    """物理情報SNNのためのトレーナー。"""
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                logits, spikes, mem_sequence = self.model(input_ids, return_spikes=True, return_full_mems=True)
                loss_dict = self.criterion(logits, target_ids, spikes, mem_sequence, self.model)
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            if self.astrocyte_network:
                self.astrocyte_network.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            if hasattr(self.criterion, "ce_loss_fn") and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index
                mask = target_ids != ignore_idx
                num_masked_elements = cast(torch.Tensor, mask).sum()
                accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class PlannerTrainer:
    """学習可能プランナーSNNのための専用トレーナー。"""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        progress_bar = tqdm(dataloader, desc=f"Planner Training Epoch {epoch}")
        
        for batch in progress_bar:
            input_ids, target_plan = [t.to(self.device) for t in batch]

            self.optimizer.zero_grad()
            
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            skill_logits, _, _ = self.model(input_ids)
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            
            assert isinstance(self.criterion, PlannerLoss)
            loss_dict = self.criterion(skill_logits, target_plan)
            loss = loss_dict['total']
            
            loss.backward()
            self.optimizer.step()
            
            progress_bar.set_postfix({"loss": loss.item()})
            
            
class BPTTTrainer:
    """
    BPTT (Backpropagation Through Time) および SLTT を用いたトレーナー。
    """

    def __init__(self, model: nn.Module, config: DictConfig):
        self.model = model
        self.config = config
        self.optimizer = Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.spike_loss = SpikeRegularizationLoss(target_rate=config.training.get("target_spike_rate", 0.02))
        self.use_sltt = self.config.training.get("use_sltt", False)
        self.model_type = self.config.model.get("type", "simple")

    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """モデルの出力形状に合わせて損失を計算する。"""
        if self.model_type == "spiking_transformer":
            # outputs: (Batch, Time, Vocab), targets: (Batch, Time)
            return self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        else: # simple SNN
            # outputs: (Time, Batch, Vocab), targets: (Batch, Time)
            # Reshape outputs to (Batch, Time, Vocab)
            outputs_reshaped = outputs.permute(1, 0, 2)
            return self.criterion(outputs_reshaped.reshape(-1, outputs_reshaped.size(-1)), targets.reshape(-1))


    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        """単一の学習ステップを実行する。"""
        self.optimizer.zero_grad()

        # Transformerは再帰的でないため、BPTTとSLTTの区別は実質的にない
        # このフラグは主に再帰的モデルのためにあるが、ここでは学習ロジックを統一
        if self.use_sltt:
            # SLTTの厳密な実装は複雑なため、ここではBPTTと同様の処理を行う
            # 概念的な分離としてif文を残す
            pass

        outputs = self.model(data)
        loss = self._calculate_loss(outputs, targets)

        # スパイク発火率の正則化項を追加
        if self.config.training.get("spike_regularization_coeff", 0.0) > 0:
            spike_regularization = self.spike_loss(self.model)
            total_loss = loss + self.config.training.spike_regularization_coeff * spike_regularization
        else:
            total_loss = loss

        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
