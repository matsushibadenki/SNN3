# snn_research/distillation/model_registry.py
# モデルレジストリ：学習済みモデルの管理
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
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
                                        'model_id' を含む必要がある。
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
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_registry(self, registry_data: List[Dict[str, Any]]):
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=4)

    def register_model(self, model_info: Dict[str, Any]):
        if 'model_id' not in model_info:
            raise ValueError("model_info must contain a 'model_id'")
        
        registry_data = self._load_registry()
        
        model_exists = False
        for i, model in enumerate(registry_data):
            if model.get('model_id') == model_info['model_id']:
                registry_data[i] = model_info
                model_exists = True
                break
        
        if not model_exists:
            registry_data.append(model_info)
            
        self._save_registry(registry_data)

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        registry_data = self._load_registry()
        for model in registry_data:
            if model.get('model_id') == model_id:
                return model
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        return self._load_registry()

    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        registry_data = self._load_registry()
        return [
            model for model in registry_data 
            if model.get("task_description") == task_description
        ]


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
        # インデックス作成のためにタスク説明を持つセットにも追加
        if 'task_description' in model_info:
            self.redis.sadd(f"{self.prefix}:task_index:{model_info['task_description']}", model_id)

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        key = self._get_key(model_id)
        model_data = self.redis.get(key)
        if model_data:
            if isinstance(model_data, bytes):
                return json.loads(model_data.decode('utf-8'))
            return json.loads(model_data)
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        model_keys = self.redis.keys(f"{self.prefix}:*")
        models = []
        for key in model_keys:
            # インデックス用のキーは無視
            if "task_index" in str(key):
                continue
            model_data = self.redis.get(key)
            if model_data:
                if isinstance(model_data, bytes):
                    models.append(json.loads(model_data.decode('utf-8')))
                else:
                    models.append(json.loads(model_data))
        return models
    
    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        model_ids = self.redis.smembers(f"{self.prefix}:task_index:{task_description}")
        models = []
        for model_id_bytes in model_ids:
            model_id = model_id_bytes.decode('utf-8')
            model_info = self.get_model_info(model_id)
            if model_info:
                models.append(model_info)
        return models
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
