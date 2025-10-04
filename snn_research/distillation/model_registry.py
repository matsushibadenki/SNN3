# snn_research/distillation/model_registry.py
# モデルレジストリ：学習済みモデルの管理
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
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
                                        'model_id' と 'task_description' を含む必要がある。
        """
        pass

    @abstractmethod
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        指定されたモデルIDの情報を取得する。

        Args:
            model_id (str): 情報を取得するモデルのID。

        Returns:
            Optional[Dict[str, Any]]: モデルの情報。見つからない場合はNone。
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

    @abstractmethod
    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        特定のタスク説明に一致するモデルを検索する。

        Args:
            task_description (str): 検索するタスクの説明。

        Returns:
            List[Dict[str, Any]]: タスク説明に一致するモデル情報のリスト。
        """
        pass

class FileModelRegistry(ModelRegistry):
    """
    JSONファイルを使用してモデルレジストリを管理するクラス。
    データ構造は Dict[task_description, List[model_info]] とする。
    """
    def __init__(self, registry_path: str = "runs/model_registry.json"):
        self.registry_path = registry_path
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _load_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _save_registry(self, registry_data: Dict[str, List[Dict[str, Any]]]):
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=4, ensure_ascii=False)

    def register_model(self, model_info: Dict[str, Any]):
        if 'model_id' not in model_info:
            raise ValueError("model_info must contain a 'model_id'")
        if 'task_description' not in model_info:
            raise ValueError("model_info must contain a 'task_description'")

        registry_data = self._load_registry()
        task = model_info['task_description']
        
        if task not in registry_data:
            registry_data[task] = []

        model_exists = False
        for i, model in enumerate(registry_data[task]):
            if model.get('model_id') == model_info['model_id']:
                registry_data[task][i] = model_info  # 情報を更新
                model_exists = True
                break
        
        if not model_exists:
            registry_data[task].append(model_info)
            
        self._save_registry(registry_data)

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        registry_data = self._load_registry()
        for task_models in registry_data.values():
            for model in task_models:
                if model.get('model_id') == model_id:
                    return model
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        registry_data = self._load_registry()
        all_models = []
        for task_models in registry_data.values():
            all_models.extend(task_models)
        return all_models

    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        registry_data = self._load_registry()
        return registry_data.get(task_description, [])


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

        if 'task_description' in model_info:
            task_key = f"{self.prefix}:task_index:{model_info['task_description']}"
            self.redis.sadd(task_key, model_id)

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        key = self._get_key(model_id)
        model_data = self.redis.get(key)
        if model_data:
            data_str = model_data.decode('utf-8') if isinstance(model_data, bytes) else model_data
            return json.loads(data_str)
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        # glob-style patterns like '*' can be inefficient in production on large DBs
        # Consider using SCAN for iteration without blocking the server
        for key in self.redis.scan_iter(f"{self.prefix}:*"):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            if "task_index" not in key_str:
                model_data = self.redis.get(key)
                if model_data:
                    data_str = model_data.decode('utf-8') if isinstance(model_data, bytes) else model_data
                    models.append(json.loads(data_str))
        return models
    
    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        task_key = f"{self.prefix}:task_index:{task_description}"
        model_ids = self.redis.smembers(task_key)
        
        models = []
        for model_id_bytes in model_ids:
            model_id = model_id_bytes.decode('utf-8')
            model_info = self.get_model_info(model_id)
            if model_info:
                models.append(model_info)
        return models
