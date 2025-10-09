# snn_research/cognitive_architecture/hierarchical_planner.py
#
# Title: 階層型プランナー
#
# Description: 高レベルの目標を、実行可能なサブタスクのシーケンスに分解するプランナー。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: 存在しない`registry`属性へのアクセスを削除。
#              mypyエラー修正: planner_modelをOptionalに変更。
#              mypyエラー修正: snn-cli.pyからの呼び出しに対応するため、execute_taskメソッドを追加。
# 改善点: ハードコードされた計画立案ロジックを、学習済みPlannerSNNを利用する形式に置き換え。
#         Tokenizerをコンストラクタで受け取るように変更。
# mypyエラー修正: .item()が返す型の曖昧さを解消するため、int()でキャストする。
# 改善点: RAGSystemを統合し、ドキュメント検索による文脈生成機能を追加。
# mypyエラー修正: asyncioをインポート。
# 改善点: ルールベースのプランニングに日本語キーワードを追加。
#
# 改善点:
# - ROADMAPフェーズ7に基づき、RAGSystemのナレッジグラフ機能を利用した
#   記号推論による計画立案を行うように修正。

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer
import asyncio

from .planner_snn import PlannerSNN
from snn_research.distillation.model_registry import ModelRegistry
from .rag_snn import RAGSystem

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
    PlannerSNNとRAGSystemを内部で利用して、動的に計画を生成する。
    """
    def __init__(
        self,
        model_registry: ModelRegistry,
        rag_system: RAGSystem,
        planner_model: Optional[PlannerSNN] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        self.model_registry = model_registry
        self.rag_system = rag_system
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

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def create_plan(self, high_level_goal: str, context: Optional[str] = None) -> Plan:
        """
        目標に基づいて計画を作成する。PlannerSNNが利用可能であればそれを使用する。
        RAGシステムのナレッジグラフを活用して、記号推論に基づいた計画を試みる。
        """
        print(f"🌍 Creating plan for goal: {high_level_goal}")

        # ステップ1: RAGのナレッジグラフ機能で関連概念を検索
        knowledge_query = f"Find concepts and relations for: {high_level_goal}"
        retrieved_knowledge = self.rag_system.search(knowledge_query, k=5)
        
        full_prompt = f"Goal: {high_level_goal}\n\nRetrieved Knowledge:\n{' '.join(retrieved_knowledge)}"
        if context:
            full_prompt += f"\n\nUser Provided Context:\n{context}"
        
        print(f"🧠 Planner is reasoning with prompt: {full_prompt[:200]}...")

        if self.planner_model:
            # --- PlannerSNNによる動的な計画生成 ---
            self.planner_model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(full_prompt, return_tensors="pt")
                input_ids = inputs['input_ids'].to(self.device)
                skill_logits, _, _ = self.planner_model(input_ids)
                predicted_skill_id = int(torch.argmax(skill_logits, dim=-1).item())
                task = self.SKILL_MAP.get(predicted_skill_id, self.SKILL_MAP[4])
                task_list = [task]
                print(f"🧠 PlannerSNN predicted skill ID: {predicted_skill_id} -> Task: {task['task']}")
        else:
            # --- フォールバック: ルールベースの簡易的な計画生成 ---
            print("⚠️ PlannerSNN model not found. Falling back to rule-based planning.")
            task_list = []
            
            # 検索された知識やゴールに基づいてタスクを決定
            prompt_lower = full_prompt.lower()
            if "summarize" in prompt_lower or "要約" in prompt_lower:
                task_list.append(self.SKILL_MAP[0])
            if "sentiment" in prompt_lower or "感情" in prompt_lower or "分析" in prompt_lower:
                task_list.append(self.SKILL_MAP[1])
            if "translate" in prompt_lower or "翻訳" in prompt_lower:
                task_list.append(self.SKILL_MAP[2])
            
            if not task_list:
                task_list.append(self.SKILL_MAP[4]) # デフォルトは汎用QA

        print(f"✅ Plan created with {len(task_list)} step(s).")
        return Plan(goal=high_level_goal, task_list=task_list)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        """
        タスク要求を受け取り、計画立案から実行までを行う。
        """
        print(f"Executing task: {task_request} with context: {context}")
        
        # 非同期メソッドを同期的に呼び出す
        plan = asyncio.run(self.create_plan(task_request, context))
        
        # ToDo: 実際にプランのタスクリストをループして専門家SNNを実行するロジックを実装
        # ここではプランの内容を返すダミー実装
        if plan.task_list:
            final_result = f"Plan for '{task_request}':\n"
            for i, task in enumerate(plan.task_list):
                final_result += f"  Step {i+1}: Execute '{task['task']}' using expert '{task['expert_id']}'.\n"
            final_result += "Task completed successfully (dummy execution)."
            return final_result
        else:
            return "Could not create a plan for the given task."
