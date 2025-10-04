# matsushibadenki/snn3/snn_research/distillation/knowledge_distillation_manager.py
# Title: 知識蒸留マネージャー
# Description: 教師モデルから生徒モデルへの知識蒸留プロセス全体を管理・実行するクラス。
# mypyエラー修正: 存在しない`get`メソッド呼び出しを修正。
# mypyエラー修正: register_modelの引数をキーワード引数に変更。
# mypyエラー修正: __init__を修正し、必要な依存関係をすべて受け取るように変更。

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List
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
        student_model: torch.nn.Module,
        trainer: DistillationTrainer,
        teacher_model_name: str,
        tokenizer_name: str,
        model_registry: ModelRegistry,
        device: str
    ):
        self.student_model = student_model.to(device)
        self.distillation_trainer = trainer
        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model_registry = model_registry
        self.device = device

    def prepare_dataset(self, texts: List[str], max_length: int, batch_size: int) -> DataLoader:
        """
        テキストデータから知識蒸留用のデータセットとデータローダーを準備する。
        """
        class _DistillationTextDataset(Dataset):
            def __init__(self, tokenizer, texts, max_length):
                self.tokenizer = tokenizer
                self.texts = texts
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                tokenized = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                return {
                    'input_ids': tokenized['input_ids'].squeeze(),
                    'attention_mask': tokenized['attention_mask'].squeeze()
                }

        dataset = _DistillationTextDataset(self.tokenizer, texts, max_length)
        return DataLoader(dataset, batch_size=batch_size)


    async def run_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_id: str,
        task_description: str,
        student_config: Dict[str, Any],
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
        save_dir = os.path.join("runs", "specialists", task_description.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        print(f"Step 3: Saving the model to {save_path}...")
        torch.save(self.student_model.state_dict(), save_path)
        print("Model saved.")

        # 4. モデルレジストリへの登録
        print("Step 4: Registering the model...")
        await self.model_registry.register_model(
            model_id=model_id,
            task_description=task_description,
            metrics=final_metrics,
            model_path=save_path,
            config=student_config
        )
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
            inputs = batch['input_ids'].to(self.device)

            with torch.no_grad():
                outputs = self.student_model(inputs, return_spikes=True)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    _, spikes, _ = outputs
                else:
                    # Handle cases where the model might not return spikes
                    spikes = torch.tensor(0.0)


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
