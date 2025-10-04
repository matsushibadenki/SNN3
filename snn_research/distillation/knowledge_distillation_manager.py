# matsushibadenki/snn3/snn_research/distillation/knowledge_distillation_manager.py
# Title: 知識蒸留マネージャー
# Description: 教師モデルから生徒モデルへの知識蒸留プロセス全体を管理・実行するクラス。
# mypyエラー修正: 存在しない`get`メソッド呼び出しを修正。
# mypyエラー修正: register_modelの引数をキーワード引数に変更。

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Any, Optional
import asyncio
import os
from tqdm import tqdm

from snn_research.training.trainers import DistillationTrainer
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.benchmark.metrics import calculate_perplexity, calculate_energy_consumption

class KnowledgeDistillationManager:
    """
    知識蒸留プロセスを統括するマネージャークラス。
    """
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        distillation_trainer: DistillationTrainer,
        model_registry: ModelRegistry,
        device: str
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.tokenizer = tokenizer
        self.distillation_trainer = distillation_trainer
        self.model_registry = model_registry
        self.device = device

    async def run_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_id: str,
        task_description: str,
        save_path: str
    ) -> Dict[str, Any]:
        """
        知識蒸留の全プロセスを実行し、学習済みモデルを登録する。
        """
        print(f"--- Starting Knowledge Distillation for model: {model_id} ---")
        
        # 1. 知識蒸留の実行
        print("Step 1: Running distillation training...")
        final_metrics = self.distillation_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            teacher_model=self.teacher_model
        )
        print("Distillation training finished.")

        # 2. モデルの評価
        print("Step 2: Evaluating the distilled model...")
        evaluation_results = await self.evaluate_model(val_loader)
        final_metrics.update(evaluation_results)
        print(f"Evaluation finished. Metrics: {final_metrics}")

        # 3. モデルの保存
        print(f"Step 3: Saving the model to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.student_model.state_dict(), save_path)
        print("Model saved.")

        # 4. モデルレジストリへの登録
        print("Step 4: Registering the model...")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        await self.model_registry.register_model(
            model_id=model_id,
            task_description=task_description,
            metrics=final_metrics,
            model_path=save_path,
            config=self.student_model.config.to_dict() if hasattr(self.student_model, 'config') else {}
        )
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        print(f"Model '{model_id}' successfully registered.")
        
        print("--- Knowledge Distillation Finished ---")
        return {"model_id": model_id, "metrics": final_metrics, "path": save_path}

    async def evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        蒸留済みモデルの性能を評価する。
        """
        self.student_model.eval()
        total_spikes = 0
        total_samples = 0
        
        # 簡易的な評価ループ
        progress_bar = tqdm(dataloader, desc="Evaluating Distilled Model")
        for batch in progress_bar:
            inputs, _, _ = batch
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                _, spikes, _ = self.student_model(inputs, return_spikes=True)
            
            total_spikes += spikes.sum().item()
            total_samples += inputs.size(0)

        avg_spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0
        
        # TODO: より正確なパープレキシティとエネルギー消費の計算
        perplexity = calculate_perplexity(self.student_model, dataloader, self.device)
        energy = calculate_energy_consumption(avg_spikes_per_sample)

        return {
            "perplexity": perplexity,
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "estimated_energy_consumption": energy
        }