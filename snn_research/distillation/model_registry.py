# matsushibadenki/snn3/snn_research/distillation/model_registry.py
# Title: モデルレジストリ
# Description: 訓練済みモデルのメタデータを管理・検索するための抽象ベースクラスと具象クラスを定義します。
#              FileModelRegistry: JSONファイルベースのレジストリ
#              RedisModelRegistry: Redisベースのレジストリ
#              mypyエラー修正: register_modelの引数を追加し、具象クラスと一致させた。
#              mypyエラー修正: Redisの非同期メソッドにawaitを追加し、json.loadsの型安全性を確保。

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import redis.asyncio as redis  # type: ignore
import asyncio

class ModelRegistry(ABC):
    """訓練済みモデルのメタデータを管理するための抽象ベースクラス。"""
    
    @abstractmethod
    def register_model(
        self,
        task_description: str,
        model_id: str,
        model_path: str,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """新しいモデルをレジストリに登録する。"""
        pass

    @abstractmethod
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """モデルIDに基づいてモデル情報を取得する。"""
        pass

    @abstractmethod
    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        """タスク記述に最も関連するモデルを検索する。"""
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """レジストリ内のすべてのモデルをリストする。"""
        pass


class FileModelRegistry(ModelRegistry):
    """JSONファイルを使用してモデルレジストリを実装するクラス。"""
    def __init__(self, registry_path: Union[str, Path]):
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """レジストリファイルをロードする。"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_registry(self) -> None:
        """レジストリをファイルに保存する。"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def register_model(
        self,
        task_description: str,
        model_id: str,
        model_path: str,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """新しいモデルをレジストリに登録する。"""
        if model_id in self.registry:
            print(f"Warning: Model ID {model_id} already exists. Overwriting.")
        
        self.registry[model_id] = {
            "task_description": task_description,
            "model_path": model_path,
            "metrics": metrics,
            "config": config
        }
        self._save_registry()
        print(f"Model '{model_id}' registered successfully to {self.registry_path}")

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """モデルIDに基づいてモデル情報を取得する。"""
        return self.registry.get(model_id)

    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        """タスク記述に最も関連するモデルを検索する（単純な部分文字列検索）。"""
        # より高度な検索（例：ベクトル検索）は将来の実装で検討
        found_models = []
        for model_id, info in self.registry.items():
            if task_description.lower() in info.get("task_description", "").lower():
                found_models.append({"model_id": model_id, **info})
        return found_models

    def list_models(self) -> List[Dict[str, Any]]:
        """レジストリ内のすべてのモデルをリストする。"""
        return [{"model_id": model_id, **info} for model_id, info in self.registry.items()]


class RedisModelRegistry(ModelRegistry):
    """Redisをバックエンドとして使用するモデルレジストリ。"""
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.model_key_prefix = "snn_model:"
        self.task_key = "snn_tasks"

    async def register_model(
        self,
        task_description: str,
        model_id: str,
        model_path: str,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """Redisに新しいモデルを非同期で登録する。"""
        model_data = {
            "task_description": task_description,
            "model_path": model_path,
            "metrics": metrics,
            "config": config
        }
        await self.redis.set(f"{self.model_key_prefix}{model_id}", json.dumps(model_data))
        await self.redis.sadd(self.task_key, task_description)
        print(f"Model '{model_id}' registered successfully to Redis.")

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Redisからモデル情報を非同期で取得する。"""
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        data = await self.redis.get(f"{self.model_key_prefix}{model_id}")
        if isinstance(data, (str, bytes)):
            return json.loads(data)
        return None
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    async def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        """タスク記述に基づいてモデルを検索する (実装は簡略化)。"""
        # この実装は全てのモデルを取得してフィルタリングするため、大規模なデータセットには非効率。
        # 本番環境ではRedis Searchなどのより高度な検索機能を利用すべき。
        all_models = await self.list_models()
        return [
            model for model in all_models
            if model.get("task_description", "").lower() == task_description.lower()
        ]

    async def list_models(self) -> List[Dict[str, Any]]:
        """Redisに登録されているすべてのモデルをリストする。"""
        model_keys = await self.redis.keys(f"{self.model_key_prefix}*")
        models = []
        for key in model_keys:
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            model_data = await self.redis.get(key)
            if isinstance(model_data, (str, bytes)):
                model_info = json.loads(model_data)
                model_info["model_id"] = key.decode('utf-8').replace(self.model_key_prefix, "")
                models.append(model_info)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        return models

    async def get_all_tasks(self) -> List[str]:
        """登録されているすべてのタスク記述をリストする。"""
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        tasks = await self.redis.smembers(self.task_key)
        return [task.decode('utf-8') for task in tasks]
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
