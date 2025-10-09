# snn_research/cognitive_architecture/emergent_system.py
#
# Title: 創発システム
#
# Description: 異なる認知コンポーネント間の相互作用を管理し、創発的な振る舞いを引き出すシステム。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: 非同期メソッド呼び出しにawaitを追加。
#              循環インポートエラー修正: TYPE_CHECKINGを使用して型ヒントのみインポートする。
#
# 改善点:
# - ROADMAPフェーズ8に基づき、エージェント間の協調行動を実装。
# - タスク失敗時にプランナーに代替案を問い合わせ、別のエージェントに
#   タスクを再割り当てするロジックを追加。
#
# 修正点:
# - mypyエラー解消のため、`random`モジュールをインポート。
# - mypyエラー解消のため、`expert_id`がNoneの場合の`in`演算子の使用を修正。

import asyncio
from typing import List, Dict, Any, TYPE_CHECKING
import random

from .global_workspace import GlobalWorkspace
from .hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry

# --- 循環インポート解消のための修正 ---
if TYPE_CHECKING:
    from snn_research.agent.autonomous_agent import AutonomousAgent


class EmergentCognitiveSystem:
    """
    複数の認知コンポーネントを統合し、協調させることで
    創発的な高次機能を実現するシステム。
    """

    def __init__(self, planner: HierarchicalPlanner, agents: List['AutonomousAgent'], global_workspace: GlobalWorkspace, model_registry: ModelRegistry):
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
        """非同期でタスク実行サイクルを処理する。協調的再計画ロジックを含む。"""
        print(f"--- Emergent System: Executing Goal: {high_level_goal} ---")

        # 1. 初期計画の作成
        plan = await self.planner.create_plan(high_level_goal)
        self.global_workspace.broadcast("plan", f"New plan created: {plan.task_list}")

        # 2. 計画の実行
        results = []
        task_queue = plan.task_list.copy()

        while task_queue:
            task = task_queue.pop(0)
            
            expert_id = task.get("expert_id")
            # expert_idに基づいてエージェントを割り当てる
            agent_name = "web_crawler_agent" if expert_id == "web_crawler" else "AutonomousAgent"
            agent = self.agents.get(agent_name)

            if not agent:
                error_msg = f"Agent '{agent_name}' for expert '{expert_id}' not found."
                results.append(error_msg)
                continue

            # エージェントがタスクを実行 (ダミー実行)
            task_description = task.get("description", "")
            print(f"-> Assigning task '{task_description}' to agent '{agent.name}'")
            # ダミー: 実行結果を模擬
            is_success = random.random() > 0.5 
            
            if is_success:
                result = f"SUCCESS: Task '{task_description}' completed by '{agent.name}'."
                results.append(result)
                self.global_workspace.broadcast(agent.name, result)
            else:
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                # --- 協調行動: タスク失敗と再計画 ---
                result = f"FAILURE: Task '{task_description}' failed by '{agent.name}'."
                results.append(result)
                self.global_workspace.broadcast(agent.name, result)
                
                print(f"!! Task failed. Attempting to find a collaborator...")
                new_task = await self.planner.refine_plan(task)
                
                if new_task:
                    print(f"++ Collaboration proposed! Re-assigning task with new expert '{new_task['expert_id']}'.")
                    task_queue.insert(0, new_task) # 新しいタスクをキューの先頭に追加
                else:
                    print("-- No collaborator found. Aborting this task branch.")
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

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
