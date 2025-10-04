# matsushibadenki/snn3/snn_research/cognitive_architecture/emergent_system.py
# Title: 創発システム
# Description: 異なる認知コンポーネント間の相互作用を管理し、創発的な振る舞いを引き出すシステム。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: 非同期メソッド呼び出しにawaitを追加。

from typing import List
import asyncio
from .global_workspace import GlobalWorkspace
from .hierarchical_planner import HierarchicalPlanner
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.distillation.model_registry import ModelRegistry

class EmergentCognitiveSystem:
    """
    複数の認知コンポーネントを統合し、協調させることで
    創発的な高次機能を実現するシステム。
    """

    def __init__(self, planner: HierarchicalPlanner, agents: List[AutonomousAgent], global_workspace: GlobalWorkspace, model_registry: ModelRegistry):
        self.planner = planner
        self.agents = {agent.name: agent for agent in agents}
        self.global_workspace = global_workspace
        self.model_registry = model_registry

    def execute_task(self, high_level_goal: str) -> str:
        """
        高レベルの目標を受け取り、計画、実行、情報統合のサイクルを実行する。
        """
        return asyncio.run(self.execute_task_async(high_level_goal))

    async def execute_task_async(self, high_level_goal: str) -> str:
        """非同期でタスク実行サイクルを処理する。"""
        print(f"--- Emergent System: Executing Goal: {high_level_goal} ---")

        # 1. 計画
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        plan = await self.planner.create_plan(high_level_goal)
        self.global_workspace.broadcast("plan", f"New plan created: {plan.task_list}")

        # 2. 実行
        results = []
        for task in plan.task_list:
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            # タスクに最適なエージェントを選択（ここでは簡略化）
            agent_name = task.get("agent", "default_agent")
            agent = self.agents.get(agent_name)

            if not agent:
                error_msg = f"Agent '{agent_name}' not found."
                print(error_msg)
                results.append(error_msg)
                continue

            # エージェントがタスクを実行
            task_description = task.get("description", "")
            result = agent.execute(task_description)
            results.append(result)

            # 結果をグローバルワークスペースにブロードキャスト
            self.global_workspace.broadcast(agent.name, result)

        # 3. 統合と要約
        final_report = self._synthesize_results(results)
        self.global_workspace.broadcast("system", f"Goal '{high_level_goal}' completed. Final report generated.")
        print(f"--- Emergent System: Goal Execution Finished ---")
        return final_report

    def _synthesize_results(self, results: List[str]) -> str:
        """
        各エージェントからの結果を統合し、最終的なレポートを生成する。
        """
        report = "Execution Summary:\n"
        for i, res in enumerate(results):
            report += f"- Step {i+1}: {res}\n"
        return report
