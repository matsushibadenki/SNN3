# matsushibadenki/snn3/snn_research/distillation/knowledge_distillation_manager.py
# Title: 知識蒸留マネージャー
# Description: 教師モデルから生徒モデルへの知識蒸留プロセス全体を管理・実行するクラス。
# mypyエラー修正: 存在しない`get`メソッド呼び出しを修正。
# mypyエラー修正: register_modelの引数をキーワード引数に変更。
# mypyエラー修正: __init__を修正し、必要な依存関係をすべて受け取るように変更。
# 改善点: run_on_demand_pipelineメソッドを新規実装。

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List
import asyncio
import os
import json
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_registry = model_registry
        self.device = device

    def prepare_dataset(self, texts: List[str], max_length: int, batch_size: int) -> DataLoader:
        """
        テキストデータから知識蒸留用のデータセットとデータローダーを準備する。
        """
        class _DistillationTextDataset(Dataset):
            def __init__(self, tokenizer, texts, max_length, teacher_model, device):
                self.tokenizer = tokenizer
                self.texts = texts
                self.max_length = max_length
                self.teacher_model = teacher_model
                self.device = device

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
                input_ids = tokenized['input_ids'].squeeze(0)
                
                # Note: 本来は事前計算が望ましいが、ここでは動的にロジットを生成
                with torch.no_grad():
                    teacher_logits = self.teacher_model(input_ids.unsqueeze(0).to(self.device)).logits.squeeze(0).cpu()
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': tokenized['attention_mask'].squeeze(),
                    'teacher_logits': teacher_logits
                }

        dataset = _DistillationTextDataset(self.tokenizer, texts, max_length, self.teacher_model, self.device)
        
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            targets = torch.roll(input_ids, shifts=-1, dims=1)
            targets[:, -1] = self.tokenizer.pad_token_id
            teacher_logits = torch.stack([item['teacher_logits'] for item in batch])
            return input_ids, targets, teacher_logits

        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


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
        # trainer.trainは存在しないため、trainerのtrain_epochとevaluateを直接呼び出す
        for epoch in range(epochs):
            self.distillation_trainer.train_epoch(train_loader, epoch)
            final_metrics = self.distillation_trainer.evaluate(val_loader, epoch)
        print("Distillation training finished.")

        # 2. モデルの評価 (最終)
        print("Step 2: Evaluating the distilled model...")
        evaluation_results = await self.evaluate_model(val_loader)
        final_metrics.update(evaluation_results)
        print(f"Evaluation finished. Metrics: {final_metrics}")

        # 3. モデルの保存
        save_dir = os.path.join("runs", "specialists", task_description.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        print(f"Step 3: Saving the model to {save_path}...")
        # DDPでラップされている可能性を考慮
        model_state_dict = self.student_model.module.state_dict() if hasattr(self.student_model, 'module') else self.student_model.state_dict()
        torch.save(model_state_dict, save_path)
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
        return {"model_id": model_id, "metrics": final_metrics, "path": save_path, "config": student_config}

    async def run_on_demand_pipeline(self, task_description: str, unlabeled_data_path: str, force_retrain: bool):
        """Webクローラー等からのデータでオンデマンド学習を実行するパイプライン。"""
        print(f"🚀 Starting on-demand pipeline for task: {task_description}")
        
        # 1. データ読み込み
        texts = []
        with open(unlabeled_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    texts.append(json.loads(line)['text'])
                except (json.JSONDecodeError, KeyError):
                    # JSONL形式でない場合も考慮
                    if line.strip():
                        texts.append(line.strip())
        
        if not texts:
            print("❌ No text found in the provided data file. Aborting.")
            return

        # 2. データローダー準備
        # ToDo: DIコンテナから設定を取得する
        max_len = 128
        batch_size = 4
        train_loader = self.prepare_dataset(texts, max_length=max_len, batch_size=batch_size)
        
        # 3. 蒸留実行
        await self.run_distillation(
            train_loader=train_loader,
            val_loader=train_loader, # 簡易的に同じデータを使用
            epochs=5, # デモ用にエポック数を設定
            model_id=task_description,
            task_description=f"Expert for {task_description}",
            student_config={} # ToDo: コンテナから取得
        )

    async def evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        蒸留済みモデルの性能を評価する。
        """
        self.student_model.eval()
        total_spikes = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc="Evaluating Distilled Model")
        for batch in progress_bar:
            # Dataloaderの出力形式に合わせて調整
            inputs, _, _ = batch
            inputs = inputs.to(self.device)

            with torch.no_grad():
                # BreakthroughTrainerのロジックを参考に修正
                model_to_eval = self.student_model.module if hasattr(self.student_model, 'module') else self.student_model
                outputs = model_to_eval(inputs, return_spikes=True)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    _, spikes, _ = outputs
                else:
                    spikes = torch.tensor(0.0)

            total_spikes += spikes.sum().item()
            total_samples += inputs.size(0)

        avg_spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0

        # ToDo: より正確なパープレキシティとエネルギー消費の計算
        perplexity = calculate_perplexity(self.student_model, dataloader, self.device)
        energy = calculate_energy_consumption(avg_spikes_per_sample)

        return {
            "perplexity": perplexity,
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "estimated_energy_consumption": energy
        }
