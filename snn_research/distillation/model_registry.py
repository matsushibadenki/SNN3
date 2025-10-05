# matsushibadenki/snn3/snn_research/distillation/model_registry.py
# ファイルパス: matsushibadenki/snn3/SNN3-176e5ceb739db651438b22d74c0021f222858011/snn_research/distillation/model_registry.py
# タイトル: モデルレジストリ
# 機能説明: モデル検索ロジックをタスク名による直接ルックアップに修正し、登録ロジックをリストに追加するよう修正。

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
from pathlib import Path

class ModelRegistry(ABC):
    # (インターフェースの変更なし)
    ...

class SimpleModelRegistry(ModelRegistry):
    """
    JSONファイルを使用したシンプルなモデルレジストリの実装。
    """
    def __init__(self, registry_path: str = "runs/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=4, ensure_ascii=False)

    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        new_model_info = {
            "task_description": task_description,
            "metrics": metrics,
            "model_path": model_path,
            "config": config
        }
        if model_id not in self.models:
            self.models[model_id] = []
        self.models[model_id].append(new_model_info)
        self._save()
        print(f"Model for task '{model_id}' registered at '{model_path}'.")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]::
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        if task_description in self.models:
            models_for_task = self.models[task_description]
            # 精度が高い順にソートする
            models_for_task.sort(
                key=lambda x: x.get("metrics", {}).get("accuracy", 0),
                reverse=True
            )
            # 後続処理で使いやすいように、各モデル情報にモデルIDを追加して返す
            for model in models_for_task:
                model['model_id'] = task_description
            return models_for_task[:top_k]
        return []
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        # 最初のモデルを返す（find_models_for_taskでソート済み）
        return self.models.get(model_id, [None])[0]

    async def list_models(self) -> List[Dict[str, Any]]:
        all_models = []
        for model_id, model_list in self.models.items():
            for model_info in model_list:
                model_info_with_id = {'model_id': model_id, **model_info}
                all_models.append(model_info_with_id)
        return all_models
