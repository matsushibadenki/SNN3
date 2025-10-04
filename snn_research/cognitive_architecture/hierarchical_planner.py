# matsushibadenki/snn3/snn_research/cognitive_architecture/hierarchical_planner.py
# Title: 階層型プランナー
# Description: 高レベルの目標を、実行可能なサブタスクのシーケンスに分解するプランナー。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: 存在しない`registry`属性へのアクセスを削除。

from typing import List, Dict, Any

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
    def __init__(self, model_registry: ModelRegistry, planner_model: PlannerSNN = None):
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.model_registry = model_registry
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.planner_model = planner_model # 将来の拡張用

    def create_plan(self, high_level_goal: str) -> Plan:
        """
        目標に基づいて計画を作成する。
        現在はルールベースだが、将来的にはPlannerSNNがこのロジックを担う。
        """
        print(f"Creating plan for goal: {high_level_goal}")
        
        # 専門家モデルのリストを取得
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        available_experts = self.model_registry.list_models()
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        # ルールベースのマッチング（デモ用）
        task_list: List[Dict[str, Any]] = []
        if "summarize" in high_level_goal and "translate" in high_level_goal:
            task_list.append({"task": "summarization", "description": "Summarize the input text.", "expert_id": "expert_summarizer_v1"})
            task_list.append({"task": "translation", "description": "Translate the summary to Japanese.", "expert_id": "expert_translator_v1"})
        elif "analyze" in high_level_goal:
            task_list.append({"task": "sentiment_analysis", "description": "Analyze the sentiment of the text.", "expert_id": "expert_sentiment_v2"})
        else:
            # デフォルトのタスク
            task_list.append({"task": "general_qa", "description": high_level_goal, "expert_id": "general_snn_v3"})

        print(f"Plan created with {len(task_list)} steps.")
        return Plan(goal=high_level_goal, task_list=task_list)
