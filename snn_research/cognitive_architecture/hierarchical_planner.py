# matsushibadenki/snn3/snn_research/cognitive_architecture/hierarchical_planner.py
# Title: 階層型プランナー
# Description: 高レベルの目標を、実行可能なサブタスクのシーケンスに分解するプランナー。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: 存在しない`registry`属性へのアクセスを削除。
#              mypyエラー修正: planner_modelをOptionalに変更。
#              mypyエラー修正: snn-cli.pyからの呼び出しに対応するため、execute_taskメソッドを追加。
# 改善点: ハードコードされた計画立案ロジックを、学習済みPlannerSNNを利用する形式に置き換え。
#         Tokenizerをコンストラクタで受け取るように変更。
# mypyエラー修正: .item()が返す型の曖昧さを解消するため、int()でキャストする。

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer

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
    PlannerSNNを内部で利用して、動的に計画を生成する。
    """
    def __init__(
        self,
        model_registry: ModelRegistry,
        planner_model: Optional[PlannerSNN] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        self.model_registry = model_registry
        self.planner_model = planner_model
        # PlannerSNNがテキストを理解するためにTokenizerが必要
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        if self.planner_model:
            self.planner_model.to(self.device)

        # ダミーのスキルリスト（実際のスキルセットに応じて要変更）
        self.SKILL_MAP: Dict[int, Dict[str, Any]] = {
            0: {"task": "summarization", "description": "Summarize the input text.", "expert_id": "expert_summarizer_v1"},
            1: {"task": "sentiment_analysis", "description": "Analyze the sentiment of the text.", "expert_id": "expert_sentiment_v2"},
            2: {"task": "translation", "description": "Translate the summary to Japanese.", "expert_id": "expert_translator_v1"},
            3: {"task": "web_search", "description": "Search the web for information.", "expert_id": "web_crawler"},
            4: {"task": "general_qa", "description": "Answer a general question.", "expert_id": "general_snn_v3"},
        }


    async def create_plan(self, high_level_goal: str) -> Plan:
        """
        目標に基づいて計画を作成する。PlannerSNNが利用可能であればそれを使用する。
        """
        print(f"🌍 Creating plan for goal: {high_level_goal}")

        if self.planner_model:
            # --- PlannerSNNによる動的な計画生成 ---
            self.planner_model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(high_level_goal, return_tensors="pt")
                input_ids = inputs['input_ids'].to(self.device)

                # PlannerSNNがスキルIDのシーケンスを予測
                skill_logits, _, _ = self.planner_model(input_ids)
                
                # 最も可能性の高いスキルを一つ選択（シーケンス予測は将来の拡張）
                predicted_skill_id_val = torch.argmax(skill_logits, dim=-1).item()
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                # mypyエラーを解消するため、int()で明示的にキャスト
                predicted_skill_id = int(predicted_skill_id_val)
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                
                # 予測されたIDからタスクを構築
                task = self.SKILL_MAP.get(predicted_skill_id)
                task_list = [task] if task else []
                
                print(f"🧠 PlannerSNN predicted skill ID: {predicted_skill_id} -> Task: {task['task'] if task else 'Unknown'}")

        else:
            # --- フォールバック: ルールベースの簡易的な計画生成 ---
            print("⚠️ PlannerSNN model not found. Falling back to rule-based planning.")
            task_list = []
            if "summarize" in high_level_goal:
                task_list.append(self.SKILL_MAP[0])
            if "analyze" in high_level_goal:
                task_list.append(self.SKILL_MAP[1])
            if "translate" in high_level_goal:
                task_list.append(self.SKILL_MAP[2])
            
            if not task_list:
                task_list.append(self.SKILL_MAP[4]) # デフォルトは汎用QA

        print(f"✅ Plan created with {len(task_list)} step(s).")
        return Plan(goal=high_level_goal, task_list=task_list)

    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        """ダミーの実装 (snn-cli.pyからの呼び出しに対応)"""
        print(f"Executing task: {task_request} with context: {context}")
        # 実際にはここでcreate_planを呼び出し、タスクリストを実行するロジックが入る
        return "Task completed successfully (dummy execution)."
