# matsushibadenki/snn3/snn_research/distillation/model_registry.py
# Title: モデルレジストリ
# Description: 専門家SNNモデルのメタデータを管理し、タスクに最適なモデルを発見するためのインターフェース。
#              mypyエラー修正: 具象クラスを追加。
#              mypyエラー修正: register_modelの引数を修正。

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
from pathlib import Path

class ModelRegistry(ABC):
    """
    専門家モデルを管理するためのインターフェース。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    @abstractmethod
    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        """新しいモデルをレジストリに登録する。"""
        pass
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    @abstractmethod
    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """特定のタスクに最適なモデルを検索する。"""
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        """モデルIDに基づいてモデル情報を取得する。"""
        pass

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """登録されているすべてのモデルのリストを取得する。"""
        pass

class SimpleModelRegistry(ModelRegistry):
    """
    JSONファイルを使用したシンプルなモデルレジストリの実装。
    """
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=4)

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        self.models[model_id] = {
            "model_id": model_id,
            "task_description": task_description,
            "metrics": metrics,
            "path": model_path,
            "config": config
        }
        self._save()
        print(f"Model '{model_id}' registered at '{model_path}'.")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        scored_models = []
        for model_id, model_info in self.models.items():
            score = 0
            for keyword in task_description.split():
                if keyword in model_info["task_description"]:
                    score += 1
            if score > 0:
                scored_models.append((score, model_info))
        
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        return [model for score, model in scored_models[:top_k]]

    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        return self.models.get(model_id)

    async def list_models(self) -> List[Dict[str, Any]]:
        return list(self.models.values())