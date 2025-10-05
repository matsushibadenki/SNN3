# matsushibadenki/snn3/snn_research/distillation/knowledge_distillation_manager.py
# タイトル: 知識蒸留マネージャー
# 機能説明: 循環インポートエラーを解消するため、型チェック時のみDistillationTrainerをインポートするように修正。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import asyncio
import os
import json
from tqdm import tqdm
from omegaconf import OmegaConf

# from snn_research.training.trainers import DistillationTrainer
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.benchmark.metrics import calculate_perplexity, calculate_energy_consumption

# --- 循環インポート解消のための修正 ---
# 型チェック時のみインポートを実行し、実行時の循環参照を回避する
if TYPE_CHECKING:
    from snn_research.training.trainers import DistillationTrainer

class KnowledgeDistillationManager:
    """
    知識蒸留プロセスを統括するマネージャークラス。
    """
    def __init__(
        self,
        student_model: torch.nn.Module,
        trainer: "DistillationTrainer",
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
        
        # ファイルパスおよびレジストリキーとして安全なIDを生成
        safe_model_id = model_id.lower().replace(" ", "_")
        print(f"--- Starting Knowledge Distillation for model: {safe_model_id} ---")


        """
        知識蒸留の全プロセスを実行し、学習済みモデルを登録する。
        """
        print(f"--- Starting Knowledge Distillation for model: {model_id} ---")
        final_metrics: Dict[str, float] = {}

        # 1. 知識蒸留の実行
        print("Step 1: Running distillation training...")
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
        # ファイルパスとして安全なIDを生成 (小文字化、スペースをアンダースコアに)
        # これにより、常に一貫したパスが生成・登録される
        save_dir = os.path.join("runs", "specialists", safe_model_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        print(f"Step 3: Saving the model to {save_path}...")
        
        # DDPでラップされている可能性を考慮し、trainerからモデルを取得
        model_to_save = self.distillation_trainer.model
        model_state_dict = model_to_save.module.state_dict() if isinstance(model_to_save, nn.parallel.DistributedDataParallel) else model_to_save.state_dict()
        torch.save(model_state_dict, save_path)
        print("Model saved.")

        # 4. モデルレジストリへの登録
        print("Step 4: Registering the model...")
        # 登録時もサニタイズされたIDをキーとして使用する
        await self.model_registry.register_model(
            model_id=safe_model_id,
            task_description=task_description,
            metrics=final_metrics,
            model_path=save_path,
            config=student_config
        )
        print(f"Model '{safe_model_id}' successfully registered.")

        print("--- Knowledge Distillation Finished ---")
        return {"model_id": safe_model_id, "metrics": final_metrics, "path": save_path, "config": student_config}

    async def run_on_demand_pipeline(self, task_description: str, unlabeled_data_path: str, force_retrain: bool, student_config: Optional[Dict[str, Any]] = None):
        """Webクローラー等からのデータでオンデマンド学習を実行するパイプライン。"""
        print(f
