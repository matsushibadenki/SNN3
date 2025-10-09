# matsushibadenki/snn3/snn_research/agent/digital_life_form.py
#
# DigitalLifeForm オーケストレーター
#
# 概要：内発的動機付けとメタ認知に基づき、各種エージェントを自律的に起動するマスタープロセス。
# mypyエラー修正: RLAgentをReinforcementLearnerAgentに修正。
# mypyエラー修正: snn-cli.pyからの呼び出しに対応するため、awareness_loopメソッドを追加し、__init__を修正。
# 改善点: 依存関係を具象クラスで解決するように修正。
#
# 改善点:
# - ROADMAP.mdのフェーズ5に基づき、PhysicsEvaluatorを導入。
# - 意思決定ロジックに物理法則（エネルギー効率、処理の滑らかさ）の評価を組み込み、
#   より高度な自律的判断を可能にした。
#
# 改善点:
# - ROADMAP.mdのフェーズ6に基づき、意思決定ロジックを確率的選択モデルに変更。
# - 複数の内部状態から各行動のスコアを算出し、重み付けされた確率で次の行動を決定する。
# - 記憶システムに多目的報酬ベクトルを記録するように修正。

import time
import logging
import torch
import random
from typing import Dict, Any
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
# 各エージェントのインポート
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import SimpleModelRegistry
from snn_research.tools.web_crawler import WebCrawler
from snn_research.cognitive_architecture.rag_snn import RAGSystem


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitalLifeForm:
    """
    内発的動機付けシステムとメタ認知SNNを統合し、
    永続的で自己駆動する学習ループを実現するオーケストレーター。
    """
    def __init__(self):
        self.motivation_system = IntrinsicMotivationSystem()
        self.meta_cognitive_snn = MetaCognitiveSNN()
        self.memory = Memory()
        self.physics_evaluator = PhysicsEvaluator()
        
        # 具象クラスで依存関係を解決
        model_registry = SimpleModelRegistry()
        rag_system = RAGSystem()
        web_crawler = WebCrawler()
        
        # 依存関係を注入してプランナーを作成
        # ToDo: 学習済みのPlannerSNNモデルをロードする仕組みを追加する
        planner = HierarchicalPlanner(
            model_registry=model_registry,
            rag_system=rag_system,
            planner_model=None # 現時点ではルールベースで動作
        )
        
        # 各種エージェントのインスタンス化
        self.autonomous_agent = AutonomousAgent(name="AutonomousAgent", planner=planner, model_registry=model_registry, memory=self.memory, web_crawler=web_crawler)
        self.rl_agent = ReinforcementLearnerAgent(input_size=10, output_size=4, device="cpu") # Dummy sizes for now
        self.self_evolving_agent = SelfEvolvingAgent(
            name="SelfEvolvingAgent",
            planner=planner,
            model_registry=model_registry,
            memory=self.memory,
            web_crawler=web_crawler
        )
        
        self.running = False
        self.state = {"last_action": None} # システムの現在の状態

    def start(self):
        """デジタル生命体の活動を開始する。"""
        self.running = True
        logging.info("DigitalLifeForm activated. Starting autonomous loop.")
        self.life_cycle()

    def stop(self):
        """デジタル生命体の活動を停止する。"""
        self.running = False
        logging.info("DigitalLifeForm deactivating.")

    def life_cycle(self):
        """
        メインの実行ループ。
        内部状態とパフォーマンス評価に基づき、自律的に行動を決定し続ける。
        """
        while self.running:
            # 1. 内部状態とパフォーマンス評価を取得
            internal_state = self.motivation_system.get_internal_state()
            performance_eval = self.meta_cognitive_snn.evaluate_performance()
            # 物理法則の一貫性を評価
            dummy_mem_sequence = torch.randn(100) # ダミーの膜電位系列
            dummy_spikes = (torch.rand(100) > 0.8).float() # ダミーのスパイク
            physical_rewards = self.physics_evaluator.evaluate_physical_consistency(dummy_mem_sequence, dummy_spikes)
            
            # 2. 状態に基づき次の行動を決定
            action = self._decide_next_action(internal_state, performance_eval, physical_rewards)
            
            # 3. 決定した行動を実行
            result, external_reward, expert_used = self._execute_action(action)

            # 4. 経験を記録 (多目的報酬ベクトルを記録)
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            reward_vector = {
                "external": external_reward,
                "physical": physical_rewards
            }
            decision_context = {"internal_state": internal_state, "performance_eval": performance_eval, "physical_rewards": physical_rewards}
            self.memory.record_experience(self.state, action, result, reward_vector, expert_used, decision_context)
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            
            # 5. システムの状態とメトリクスを更新
            # ToDo: 実行結果から実際の値を取得するように修正
            dummy_prediction_error = result.get("prediction_error", 0.1)
            dummy_success_rate = result.get("success_rate", 0.9)
            dummy_task_similarity = 0.8 # 実際にはタスク間の類似度を計算
            dummy_loss = result.get("loss", 0.05)
            dummy_time = result.get("computation_time", 1.0)
            dummy_accuracy = result.get("accuracy", 0.95)

            self.motivation_system.update_metrics(dummy_prediction_error, dummy_success_rate, dummy_task_similarity, dummy_loss)
            self.meta_cognitive_snn.update_metadata(dummy_loss, dummy_time, dummy_accuracy)
            self.state = {"last_action": action, "last_result": result}
            
            logging.info(f"Action: {action}, Result: {result}, Reward: {external_reward}")
            logging.info(f"New Internal State: {self.motivation_system.get_internal_state()}")
            
            time.sleep(10) # 実行間隔

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _decide_next_action(self, internal_state: Dict[str, float], performance_eval: Dict[str, Any], physical_rewards: Dict[str, float]) -> str:
        """
        状態遷移ロジック。内部状態に基づき、各行動のスコアを計算し、確率的に次の行動を選択する。
        """
        action_scores = {
            "acquire_new_knowledge": 0.0,
            "evolve_architecture": 0.0,
            "explore_new_task_with_rl": 0.0,
            "plan_and_execute": 0.0,
            "practice_skill_with_rl": 0.0,
        }

        # --- スコア計算ロジック ---
        # 知識不足の場合、知識獲得の優先度を大幅に上げる
        if performance_eval["status"] == "knowledge_gap":
            action_scores["acquire_new_knowledge"] += 10.0
            logging.info("Decision reason: Knowledge gap detected.")
        
        # 能力不足、またはエネルギー効率が悪い場合、自己進化の優先度を上げる
        if performance_eval["status"] == "capability_gap":
            action_scores["evolve_architecture"] += 5.0
            logging.info("Decision reason: Capability gap detected.")
        if physical_rewards.get("sparsity_reward", 1.0) < 0.5:
            action_scores["evolve_architecture"] += 8.0
            logging.info("Decision reason: Low energy efficiency (sparsity).")

        # 好奇心は新しいタスクの探求や複雑な計画を促進
        action_scores["explore_new_task_with_rl"] += internal_state["curiosity"] * 5.0
        action_scores["plan_and_execute"] += internal_state["curiosity"] * 3.0
        
        # 退屈している場合、新しいタスクの探求を強く推奨
        if internal_state["boredom"] > 0.7:
            action_scores["explore_new_task_with_rl"] += internal_state["boredom"] * 10.0
            logging.info("Decision reason: High boredom.")

        # 自信がある場合、既存スキルの練習（活用）よりは新しい挑戦を優先
        action_scores["practice_skill_with_rl"] += internal_state["confidence"] * 2.0
        action_scores["explore_new_task_with_rl"] += internal_state["confidence"] * 1.0

        # デフォルト行動として、常に練習には小さなスコアを与える
        action_scores["practice_skill_with_rl"] += 1.0

        # --- 確率的選択 ---
        actions = list(action_scores.keys())
        scores = [max(0, s) for s in action_scores.values()] # スコアは非負
        total_score = sum(scores)

        if total_score == 0:
            chosen_action = "practice_skill_with_rl" # スコアが全て0ならデフォルト
        else:
            probabilities = [s / total_score for s in scores]
            chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        
        logging.info(f"Action scores: {action_scores}")
        logging.info(f"Probabilities: { {a: f'{p:.2%}' for a, p in zip(actions, probabilities) if total_score > 0} }")
        logging.info(f"Chosen action: {chosen_action}")

        return chosen_action
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def _execute_action(self, action):
        """
        指定されたアクションに対応するエージェントを実行する。
        """
        try:
            if action == "acquire_new_knowledge":
                result_str = self.autonomous_agent.learn_from_web("latest SNN research trends")
                return {"status": "success", "info": result_str, "accuracy": 0.96}, 0.8, ["web_crawler"]
            elif action == "evolve_architecture":
                # ToDo: evolveメソッドの具体的な実装と連携
                result_str = self.self_evolving_agent.evolve()
                return {"status": "success", "info": result_str, "accuracy": 0.97}, 0.9, ["self_evolver"]
            elif action == "explore_new_task_with_rl":
                # ToDo: RLエージェントの具体的なタスク実行ロジックと連携
                return {"status": "success", "info": "Exploration finished.", "accuracy": 0.92}, 0.7, ["rl_agent_explorer"]
            elif action == "practice_skill_with_rl":
                # ToDo: RLエージェントの具体的なタスク実行ロジックと連携
                return {"status": "success", "info": "Practice finished.", "accuracy": 0.98}, 0.5, ["rl_agent_practicer"]
            elif action == "plan_and_execute":
                result_str = self.autonomous_agent.execute("Summarize the latest trends in SNN and analyze the sentiment.")
                return {"status": "success", "info": result_str, "accuracy": 0.95}, 0.8, ["planner", "summarizer_snn", "sentiment_snn"]
            else:
                return {"status": "failed", "info": "Unknown action"}, 0.0, []
        except Exception as e:
            logging.error(f"Error executing action '{action}': {e}")
            return {"status": "error", "info": str(e)}, -1.0, []

    def awareness_loop(self, cycles: int):
        """
        snn-cliから呼び出されるための簡易的な実行ループ。
        """
        print(f"🧬 Digital Life Form awareness loop starting for {cycles} cycles.")
        self.running = True
        for i in range(cycles):
            if not self.running:
                break
            print(f"\n----- Cycle {i+1}/{cycles} -----")
            # life_cycleは内部でループするため、ここでは1ステップ分の処理を直接呼び出す
            internal_state = self.motivation_system.get_internal_state()
            performance_eval = self.meta_cognitive_snn.evaluate_performance()
            dummy_mem_sequence = torch.randn(100)
            dummy_spikes = (torch.rand(100) > 0.8).float()
            physical_rewards = self.physics_evaluator.evaluate_physical_consistency(dummy_mem_sequence, dummy_spikes)
            action = self._decide_next_action(internal_state, performance_eval, physical_rewards)
            result, reward, expert_used = self._execute_action(action)
            
            reward_vector = {"external": reward, "physical": physical_rewards}
            decision_context = {"internal_state": internal_state, "performance_eval": performance_eval, "physical_rewards": physical_rewards}
            self.memory.record_experience(self.state, action, result, reward_vector, expert_used, decision_context)
            
            # 状態更新のダミー処理
            self.motivation_system.update_metrics(0.1, 0.9, 0.8, 0.05)
            self.meta_cognitive_snn.update_metadata(0.05, 1.0, 0.95)
            self.state = {"last_action": action, "last_result": result}
            
            logging.info(f"Action: {action}, Result: {result}, Reward: {reward}")
            time.sleep(2) # サイクル間の待機
        self.stop()
        print("🧬 Awareness loop finished.")
