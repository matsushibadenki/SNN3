# snn_research/distillation/model_registry.py
# モデルレジストリ：学習済みモデルの管理

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import redis

class ModelRegistry(ABC):
    """
    学習済みモデルの情報を管理するための抽象基底クラス。
    """

    @abstractmethod
    def register_model(self, model_info: Dict[str, Any]):
        """
        新しいモデルの情報をレジストリに登録する。

        Args:
            model_info (Dict[str, Any]): 登録するモデルの情報。
                                        'model_id' を含む必要がある。
        """
        pass

    @abstractmethod
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        指定されたモデルIDの情報を取得する。

        Args:
            model_id (str): 情報を取得するモデルのID。

        Returns:
            Dict[str, Any]: モデルの情報。
        """
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        登録されているすべてのモデルのリストを取得する。

        Returns:
            List[Dict[str, Any]]: すべてのモデル情報のリスト。
        """
        pass

class FileModelRegistry(ModelRegistry):
    """
    JSONファイルを使用してモデルレジストリを管理するクラス。
    """
    def __init__(self, registry_path: str = "runs/model_registry.json"):
        self.registry_path = registry_path
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump([], f)

    def _load_registry(self) -> List[Dict[str, Any]]:
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def _save_registry(self, registry_data: List[Dict[str, Any]]):
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=4)

    def register_model(self, model_info: Dict[str, Any]):
        if 'model_id' not in model_info:
            raise ValueError("model_info must contain a 'model_id'")
        
        registry_data = self._load_registry()
        
        # 同じ model_id があれば更新、なければ追加
        model_exists = False
        for i, model in enumerate(registry_data):
            if model.get('model_id') == model_info['model_id']:
                registry_data[i] = model_info
                model_exists = True
                break
        
        if not model_exists:
            registry_data.append(model_info)
            
        self._save_registry(registry_data)

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        registry_data = self._load_registry()
        for model in registry_data:
            if model.get('model_id') == model_id:
                return model
        raise ValueError(f"Model with id '{model_id}' not found.")

    def list_models(self) -> List[Dict[str, Any]]:
        return self._load_registry()


class RedisModelRegistry(ModelRegistry):
    """
    Redisを使用してモデルレジストリを管理するクラス。
    """
    def __init__(self, redis_client: redis.Redis, prefix: str = "snn_model"):
        self.redis = redis_client
        self.prefix = prefix

    def _get_key(self, model_id: str) -> str:
        return f"{self.prefix}:{model_id}"

    def register_model(self, model_info: Dict[str, Any]):
        if 'model_id' not in model_info:
            raise ValueError("model_info must contain a 'model_id'")
        
        model_id = model_info['model_id']
        key = self._get_key(model_id)
        self.redis.set(key, json.dumps(model_info))

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        key = self._get_key(model_id)
        model_data = self.redis.get(key)
        if model_data:
            return json.loads(model_data)
        raise ValueError(f"Model with id '{model_id}' not found.")

    def list_models(self) -> List[Dict[str, Any]]:
        model_keys = self.redis.keys(f"{self.prefix}:*")
        models = []
        for key in model_keys:
            model_data = self.redis.get(key)
            if model_data:
                models.append(json.loads(model_data))
        return models
