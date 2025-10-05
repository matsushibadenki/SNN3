# matsushibadenki/snn3/snn_research/distillation/model_registry.py
# ファイルパス: matsushibadenki/snn3/SNN3-176e5ceb739db651438b22d74c_0021f222858011/snn_research/distillation/model_registry.py
# タイトル: モデルレジストリ
# 機能説明: find_models_for_taskメソッドの末尾にあった余分なコロンを削除し、SyntaxErrorを修正。

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
from pathlib import Path

class ModelRegistry(ABC):
    """
    専門家モデルを管理するためのインターフェース。
    """
    @abstractmethod
    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        """新しいモデルをレジストリに登録する。"""
        pass

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
    def __init__(self, registry_path: str = "runs/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.project_root = self.registry_path.resolve().parent.parent
        self.models: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    # ファイルが空の場合にエラーにならないようにする
                    content = f.read()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                # ファイルが破損している、または見つからない場合は空のレジストリを返す
                return {}
        return {}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=4, ensure_ascii=False)

    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
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

    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        if task_description in self.models:
            models_for_task = self.models[task_description]
            
            models_for_task.sort(
                key=lambda x: x.get("metrics", {}).get("accuracy", 0),
                reverse=True
            )

            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            resolved_models = []
            for model_info in models_for_task[:top_k]:
                relative_path_str = model_info.get('model_path') or model_info.get('path')
                
                if relative_path_str:
                    # スクリプト実行ディレクトリからの相対パスを解決して絶対パスに変換
                    absolute_path = Path(relative_path_str).resolve()
                    model_info['model_path'] = str(absolute_path)

                model_info['model_id'] = task_description
                resolved_models.append(model_info)
            
            return resolved_models
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        return []

    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        models = self.models.get(model_id)
        if models:
            model_info = models[0] 
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            relative_path_str = model_info.get('model_path') or model_info.get('path')
            if relative_path_str:
                absolute_path = Path(relative_path_str).resolve()
                model_info['model_path'] = str(absolute_path)
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            return model_info
        return None

    async def list_models(self) -> List[Dict[str, Any]]:
        all_models = []
        for model_id, model_list in self.models.items():
            for model_info in model_list:
                model_info_with_id = {'model_id': model_id, **model_info}
                all_models.append(model_info_with_id)
        return all_models
