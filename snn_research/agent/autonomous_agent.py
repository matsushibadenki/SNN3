# matsushibadenki/snn3/snn_research/agent/autonomous_agent.py
# Title: 自律エージェント
# Description: 独自の目標を持ち、計画に基づいてタスクを実行できるエージェントの基本クラス。
#              mypyエラー修正: Memory.add_experienceをrecord_experienceに修正し、引数を適合。
#              Web学習失敗時のステータスをFAILUREとして記録するように修正。

from typing import Dict, Any
import asyncio

from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.tools.web_crawler import WebCrawler
from .memory import Memory as AgentMemory

class AutonomousAgent:
    """
    自律的にタスクを実行するエージェントのベースクラス。
    """
    def __init__(self, name: str, planner: HierarchicalPlanner, model_registry: ModelRegistry, memory: AgentMemory, web_crawler: WebCrawler):
        self.name = name
        self.planner = planner
        self.model_registry = model_registry
        self.memory = memory
        self.web_crawler = web_crawler
        self.current_state = {"agent_name": name} # 初期状態

    def execute(self, task_description: str) -> str:
        """
        与えられたタスクを実行する。
        """
        print(f"Agent '{self.name}' is executing task: {task_description}")

        if "research" in task_description or "latest information" in task_description:
            return self.learn_from_web(task_description)

        expert = asyncio.run(self.find_expert(task_description))
        action = "execute_task_with_expert" if expert else "execute_task_general"
        expert_id = [expert['model_id']] if expert else []

        if expert:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using expert model '{expert['model_id']}'."
        else:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using general capabilities (no specific expert found)."
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.memory.record_experience(
            state=self.current_state,
            action=action,
            result={"status": "SUCCESS", "details": result},
            reward=1.0,
            expert_used=expert_id,
            decision_context={"reason": "Direct execution command received."}
        )
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        return result

    async def find_expert(self, task_description: str) -> Dict[str, Any] | None:
        """
        タスクに最適な専門家モデルをモデルレジストリから検索する。
        """
        experts = await self.model_registry.find_models_for_task(task_description)
        if not experts:
            return None
        return experts[0]

    def learn_from_web(self, topic: str) -> str:
        """
        Webクローラーを使って情報を収集し、知識を更新する。
        """
        print(f"Agent '{self.name}' is learning about '{topic}' from the web.")
        urls = self._search_for_urls(topic)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        task_name = f"learn_from_web: {topic}"
        if not urls:
            result_details = "Could not find relevant information on the web."
            self.memory.record_experience(
                state=self.current_state, action=task_name,
                result={"status": "FAILURE", "details": result_details},
                reward=-1.0, expert_used=["web_crawler"],
                decision_context={"reason": "No relevant URLs found."}
            )
            return result_details

        content = self.web_crawler.crawl(urls[0])
        summary = self._summarize(content)

        self.memory.record_experience(
            state=self.current_state, action=task_name,
            result={"status": "SUCCESS", "summary": summary},
            reward=1.0, expert_used=["web_crawler", "summarizer"],
            decision_context={"reason": "Information successfully retrieved and summarized."}
        )
        return f"Successfully learned about '{topic}'. Summary: {summary}"
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def _search_for_urls(self, query: str) -> list[str]:
        # TODO: 実際の検索エンジンAPIに置き換える
        return [f"https://example.com/search?q={query.replace(' ', '+')}"]

    def _summarize(self, text: str) -> str:
        # TODO: より高度な要約モデルに置き換える
        return text[:150] + "..."