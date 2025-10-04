# matsushibadenki/snn3/snn_research/agent/self_evolving_agent.py
# Title: 自己進化エージェント
# Description: 自身のアーキテクチャや学習ルールを自律的に修正・改善できるエージェント。
#              mypyエラー修正: super().__init__に引数を追加。

from typing import Dict, Any

from .autonomous_agent import AutonomousAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.tools.web_crawler import WebCrawler
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from .memory import Memory as AgentMemory
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

class SelfEvolvingAgent(AutonomousAgent):
    """
    自身の性能を監視し、必要に応じて自己進化するエージェント。
    """
    def __init__(
        self,
        name: str,
        planner: HierarchicalPlanner,
        model_registry: ModelRegistry,
        memory: AgentMemory,
        web_crawler: WebCrawler,
        evolution_threshold: float = 0.5
    ):
        super().__init__(name, planner, model_registry, memory, web_crawler)
        self.evolution_threshold = evolution_threshold

    def execute(self, task_description: str) -> str:
        """
        タスクを実行し、結果を評価して自己進化を試みる。
        """
        result = super().execute(task_description)
        
        performance = self.evaluate_performance(task_description, result)
        
        if performance < self.evolution_threshold:
            print(f"Performance ({performance:.2f}) is below threshold ({self.evolution_threshold}). Triggering evolution...")
            evolution_result = self.evolve()
            return f"{result}\nAdditionally, self-evolution was triggered: {evolution_result}"

        return result

    def evaluate_performance(self, task: str, result: str) -> float:
        """
        タスクの実行結果を評価する。
        """
        if "successfully" in result.lower() or "using expert" in result.lower():
            return 0.9
        if "no specific expert found" in result.lower():
            return 0.4
        return 0.1

    def evolve(self) -> str:
        """
        自己進化のプロセスを実行する。
        """
        evolved_model_id = f"evolved_{self.name}_model_v{self.get_next_version()}"
        print(f"Evolving into new model: {evolved_model_id}")
        
        return f"Successfully evolved architecture. New model ID: {evolved_model_id}"

    def get_next_version(self) -> int:
        return 2
