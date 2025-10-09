# matsushibadenki/snn3/snn_research/cognitive_architecture/hierarchical_planner.py
#
# Title: 階層型プランナー
#
# 改善点:
# - ROADMAPフェーズ8に基づき、協調的タスク解決のための`refine_plan`メソッドを実装。
# - タスク失敗時に、代替となる専門家（協力者）を提案する機能を追加。

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

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def refine_plan(self, failed_task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        失敗したタスクの代替案（協力者）を提案する。
        """
        task_desc = failed_task.get("description", "")
        print(f"🤔 Refining plan for failed task: {task_desc}")

        # モデルレジストリで、同じタスクを解決できる別の専門家を検索
        alternative_experts = await self.model_registry.find_models_for_task(task_desc, top_k=5)

        # 元の専門家以外の候補を探す
        original_expert_id = failed_task.get("expert_id")
        for expert in alternative_experts:
            if expert.get("model_id") != original_expert_id:
                print(f"✅ Found alternative expert: {expert['model_id']}")
                # 新しいタスク定義を作成して返す
                new_task = failed_task.copy()
                new_task["expert_id"] = expert["model_id"]
                new_task["description"] = expert["task_description"]
                return new_task
        
        print("❌ No alternative expert found.")
        return None
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
