# matsushibadenki/snn3/snn_research/cognitive_architecture/global_workspace.py
# Title: グローバルワークスペース
# Description: 異なる認知モジュール間で情報を共有・統合するための中央ハブ。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。

from typing import Dict, Any, List, Callable

from snn_research.distillation.model_registry import ModelRegistry

class GlobalWorkspace:
    """
    認知アーキテクチャ全体で情報を共有するための中央情報ハブ。
    """
    def __init__(self, model_registry: ModelRegistry):
        self.blackboard: Dict[str, Any] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.model_registry = model_registry
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def broadcast(self, source: str, data: Any) -> None:
        """
        情報をブラックボードに書き込み、購読者に通知する。
        """
        print(f"[GlobalWorkspace] Broadcast from '{source}': {str(data)[:100]}...")
        self.blackboard[source] = data
        self._notify(source)

    def subscribe(self, source: str, callback: Callable) -> None:
        """
        特定のソースからの情報更新を購読する。
        """
        if source not in self.subscribers:
            self.subscribers[source] = []
        self.subscribers[source].append(callback)

    def _notify(self, source: str) -> None:
        """
        更新があったソースの購読者に通知する。
        """
        if source in self.subscribers:
            for callback in self.subscribers[source]:
                callback(self.blackboard[source])

    def get_information(self, source: str) -> Any:
        """
        ブラックボードから特定の情報を取得する。
        """
        return self.blackboard.get(source)

    def get_full_context(self) -> Dict[str, Any]:
        """
        現在のワークスペースの全コンテキストを取得する。
        """
        return self.blackboard.copy()
