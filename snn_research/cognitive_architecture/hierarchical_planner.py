# matsushibadenki/snn3/snn_research/cognitive_architecture/hierarchical_planner.py
# Title: 階層型プランナー
# Description: 高レベルの目標を、実行可能なサブタスクのシーケンスに分解するプランナー。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: 存在しない`registry`属性へのアクセスを削除。
#              mypyエラー修正: planner_modelをOptionalに変更。

from typing import List, Dict, Any, Optional

from .planner_snn import PlannerSNN
from snn_research.distillation.model_registry import ModelRegistry

class Plan:
    """
    タスクのシーケンスを表現するクラス。
    """
    def __init__(self, goal: str, task_list: List[Dict[str, Any]]):
        self.goal = goal
        self.task_list = task_list

    def __repr__(self) -> str:
        return f"Plan(goal='{self.goal}', tasks={len(self.task_list)})"


class HierarchicalPlanner:
    """
    高レベルの目標をサブタスクに分解する階層型プランナー。
    将来的にはPlannerSNNを内部で利用する。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, model_registry: ModelRegistry, planner_model: Optional[PlannerSNN] = None):
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.model_registry = model_registry
        self.planner_model = planner_model

    async def create_plan(self, high_level_goal: str) -> Plan:
        """
        目標に基づいて計画を作成する。
        """
        print(f"Creating plan for goal: {high_level_goal}")
        
        available_experts = await self.model_registry.list_models()

        task_list: List[Dict[str, Any]] = []
        if "summarize" in high_level_goal and "translate" in high_level_goal:
            task_list.append({"task": "summarization", "description": "Summarize the input text.", "expert_id": "expert_summarizer_v1"})
            task_list.append({"task": "translation", "description": "Translate the summary to Japanese.", "expert_id": "expert_translator_v1"})
        elif "analyze" in high_level_goal:
            task_list.append({"task": "sentiment_analysis", "description": "Analyze the sentiment of the text.", "expert_id": "expert_sentiment_v2"})
        else:
            task_list.append({"task": "general_qa", "description": high_level_goal, "expert_id": "general_snn_v3"})

        print(f"Plan created with {len(task_list)} steps.")
        return Plan(goal=high_level_goal, task_list=task_list)