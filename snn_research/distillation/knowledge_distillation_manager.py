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
        
        safe_model_id = model_id.lower().replace(" ", "_")
        print(f"--- Starting Knowledge Distillation for model: {safe_model_id} ---")

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
        save_dir = os.path.join("runs", "specialists", safe_model_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        print(f"Step 3: Saving the model to {save_path}...")
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 'mem' を含むバッファを除外して、モデルの「重み」のみを保存する
        model_to_save = self.distillation_trainer.model.module if isinstance(self.distillation_trainer.model, nn.parallel.DistributedDataParallel) else self.distillation_trainer.model
        buffers_to_exclude = {name for name, _ in model_to_save.named_buffers() if 'mem' in name}
        model_state_to_save = {k: v for k, v in model_to_save.state_dict().items() if k not in buffers_to_exclude}
        torch.save(model_state_to_save, save_path)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        print("Model saved.")

        # 4. モデルレジストリへの登録
        print("Step 4: Registering the model...")
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
        print(f"🚀 Starting on-demand pipeline for task: {task_description}")

        if student_config is None:
            print("student_config not provided, attempting to retrieve from student model...")
            if hasattr(self.student_model, 'config') and hasattr(self.student_model.config, 'model'):
                student_config = OmegaConf.to_container(self.student_model.config.model, resolve=True)
                print("✅ Successfully retrieved config from SNNCore model.")
            else:
                raise ValueError("student_config was not provided and could not be retrieved from the model.")
        
        texts = []
        with open(unlabeled_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    texts.append(json.loads(line)['text'])
                except (json.JSONDecodeError, KeyError):
                    if line.strip():
                        texts.append(line.strip())
        
        if not texts:
            print("❌ No text found in the provided data file. Aborting.")
            return

        max_len = student_config.get("time_steps", 128) if student_config and isinstance(student_config, dict) else 128
        batch_size = 4
        train_loader = self.prepare_dataset(texts, max_length=max_len, batch_size=batch_size)
        
        await self.run_distillation(
            train_loader=train_loader,
            val_loader=train_loader,
            epochs=15,
            model_id=task_description,
            task_description=f"Expert for {task_description}",
            student_config=student_config
        )

    async def evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        蒸留済みモデルの性能を評価する。
        """
        model_to_eval = self.distillation_trainer.model
        model_to_eval.eval()
        total_spikes = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc="Evaluating Distilled Model")
        for batch in progress_bar:
            inputs, _, _ = batch
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = model_to_eval(inputs, return_spikes=True)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    _, spikes, _ = outputs
                else:
                    spikes = torch.zeros((), device=inputs.device)

            total_spikes += spikes.sum().item()
            total_samples += inputs.size(0)

        avg_spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0

        perplexity = calculate_perplexity(model_to_eval, dataloader, self.device)
        energy = calculate_energy_consumption(avg_spikes_per_sample)

        return {
            "perplexity": perplexity,
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "estimated_energy_consumption": energy
        }
