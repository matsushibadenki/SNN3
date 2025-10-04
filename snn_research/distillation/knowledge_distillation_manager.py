# matsushibadenki/snn3/snn_research/distillation/knowledge_distillation_manager.py
# Title: 知識蒸留マネージャー
# Description: 知識蒸留プロセス全体を統括するクラス。
#              - 教師モデルの選択
#              - データセットの準備
#              - 生徒モデル（SNN）の訓練
#              - 訓練済みモデルのレジストリへの登録
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: register_modelの引数を基底クラスと一致させた。

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # type: ignore
from typing import Dict, Any, Optional

from snn_research.core.snn_core import BreakthroughSNN
from snn_research.training.trainers import DistillationTrainer
from snn_research.distillation.model_registry import ModelRegistry

class KnowledgeDistillationManager:
    """知識蒸留プロセスを管理するクラス。"""
    def __init__(
        self,
        student_model: BreakthroughSNN,
        trainer: DistillationTrainer,
        teacher_model_name: str,
        tokenizer_name: str,
        model_registry: ModelRegistry, # 修正: 具象クラスをDI
        device: str = "cpu"
    ):
        self.student_model = student_model.to(device)
        self.trainer = trainer
        self.teacher_model_name = teacher_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.model_registry = model_registry
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.device = device
        self.teacher_model: Optional[AutoModelForCausalLM] = None
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.student_model.config.pad_token_id = self.tokenizer.pad_token_id


    def _load_teacher_model(self) -> None:
        """教師モデルをロードする。"""
        print(f"Loading teacher model: {self.teacher_model_name}...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(self.teacher_model_name).to(self.device)
        self.teacher_model.eval()
        print("Teacher model loaded successfully.")

    def prepare_dataset(self, texts: list[str], max_length: int, batch_size: int) -> DataLoader:
        """教師モデルのロジットをラベルとしてデータセットを準備する。"""
        if self.teacher_model is None:
            self._load_teacher_model()
        
        assert self.teacher_model is not None, "Teacher model is not loaded."

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        print("Generating teacher logits...")
        with torch.no_grad():
            outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = outputs.logits.detach()
        print("Teacher logits generated.")
        
        dataset = TensorDataset(input_ids, attention_mask, teacher_logits)
        return DataLoader(dataset, batch_size=batch_size)

    def run_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_id: str,
        task_description: str,
        student_config: Dict[str, Any]
    ) -> str:
        """蒸留プロセスを実行し、訓練済みモデルを登録する。"""
        print(f"Starting knowledge distillation for model '{model_id}'...")
        
        # 訓練の実行
        final_metrics = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            teacher_model=self.teacher_model
        )
        
        # モデルの保存
        output_dir = f"runs/distilled_models/{model_id}"
        self.student_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Distillation finished. Model saved to {output_dir}")
        
        # モデルレジストリへの登録
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.model_registry.register_model(
            model_id=model_id,
            task_description=task_description,
            model_path=output_dir,
            metrics=final_metrics,
            config=student_config
        )
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        return output_dir
