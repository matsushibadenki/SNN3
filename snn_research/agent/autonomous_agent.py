# matsushibadenki/snn3/snn_research/agent/autonomous_agent.py
# Title: 自律エージェント
# Description: 独自の目標を持ち、計画に基づいてタスクを実行できるエージェントの基本クラス。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。

from typing import Dict, Any

from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.tools.web_crawler import WebCrawler
from .memory import AgentMemory

class AutonomousAgent:
    """
    自律的にタスクを実行するエージェントのベースクラス。
    """
    def __init__(self, name: str, planner: HierarchicalPlanner, model_registry: ModelRegistry, memory: AgentMemory, web_crawler: WebCrawler):
        self.name = name
        self.planner = planner
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.model_registry = model_registry
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.memory = memory
        self.web_crawler = web_crawler

    def execute(self, task_description: str) -> str:
        """
        与えられたタスクを実行する。
        この基本クラスでは、タスクをシミュレートするだけ。
        サブクラスで具体的なロジックを実装する。
        """
        print(f"Agent '{self.name}' is executing task: {task_description}")

        # Webからの学習が必要か判断（デモ用ロジック）
        if "research" in task_description or "latest information" in task_description:
            return self.learn_from_web(task_description)

        # 専門家モデルを検索して利用
        expert = self.find_expert(task_description)
        if expert:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using expert model '{expert['model_id']}'."
        else:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using general capabilities (no specific expert found)."
        
        self.memory.add_experience(task_description, result, "success")
        return result

    def find_expert(self, task_description: str) -> Dict[str, Any] | None:
        """
        タスクに最適な専門家モデルをモデルレジストリから検索する。
        """
        experts = self.model_registry.find_models_for_task(task_description)
        if not experts:
            return None
        # 最も性能の良いモデルを選択するロジック（ここでは簡略化）
        return experts[0]

    def learn_from_web(self, topic: str) -> str:
        """
        Webクローラーを使って情報を収集し、知識を更新する。
        """
        print(f"Agent '{self.name}' is learning about '{topic}' from the web.")
        urls = self._search_for_urls(topic)
        if not urls:
            return "Could not find relevant information on the web."

        content = self.web_crawler.crawl(urls[0]) # 最初のURLだけクロール
        
        # 実際には、この内容を使って新しい専門家を訓練したり、
        # RAGの知識ベースを更新したりする。
        summary = self._summarize(content)
        
        self.memory.add_experience(f"learn from web: {topic}", summary, "success")
        return f"Successfully learned about '{topic}'. Summary: {summary}"

    def _search_for_urls(self, query: str) -> list[str]:
        # ダミー実装。実際にはGoogle Search APIなどを利用する。
        return [f"https://example.com/search?q={query.replace(' ', '+')}"]

    def _summarize(self, text: str) -> str:
        # ダミー実装。実際には要約モデルを利用する。
        return text[:150] + "..."
