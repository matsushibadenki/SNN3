# matsushibadenki/snn3/snn_research/agent/digital_life_form.py
# DigitalLifeForm オーケストレーター
# 概要：内発的動機付けとメタ認知に基づき、各種エージェントを自律的に起動するマスタープロセス。
# mypyエラー修正: RLAgentをReinforcementLearnerAgentに修正。
# mypyエラー修正: snn-cli.pyからの呼び出しに対応するため、awareness_loopメソッドを追加し、__init__を修正。
# 改善点: 依存関係を具象クラスで解決するように修正。

import time
import logging
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
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
            
            # 2. 状態に基づき次の行動を決定
            action = self._decide_next_action(internal_state, performance_eval)
            
            # 3. 決定した行動を実行
            result, reward, expert_used = self._execute_action(action)

            # 4. 経験を記録
            decision_context = {"internal_state": internal_state, "performance_eval": performance_eval}
            self.memory.record_experience(self.state, action, result, reward, expert_used, decision_context)
            
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
            
            logging.info(f"Action: {action}, Result: {result}, Reward: {reward}")
            logging.info(f"New Internal State: {self.motivation_system.get_internal_state()}")
            
            time.sleep(10) # 実行間隔

    def _decide_next_action(self, internal_state, performance_eval):
        """
        状態遷移ロジック。内部状態とパフォーマンス評価から次の行動を決定する。
        """
        logging.info(f"Decision-making based on: \n- Internal State: {internal_state} \n- Performance Eval: {performance_eval}")
        
        if performance_eval["status"] == "knowledge_gap":
            logging.info("Reason: Knowledge gap detected. Acquiring new information.")
            return "acquire_new_knowledge"
        
        if performance_eval["status"] == "capability_gap":
            logging.info("Reason: Capability gap detected. Evolving model architecture.")
            return "evolve_architecture"

        if internal_state["boredom"] > 0.7 and internal_state["confidence"] > 0.8:
            logging.info("Reason: High boredom and confidence. Exploring new tasks.")
            return "explore_new_task_with_rl"
            
        if internal_state["curiosity"] > 0.6:
            logging.info("Reason: High curiosity. Planning complex task.")
            return "plan_and_execute"

        logging.info("Reason: Default behavior. Practicing existing skills.")
        return "practice_skill_with_rl"

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
            print(f"\n----- Cycle {i+1}/{cycles} -----")
            self.life_cycle()
            if not self.running:
                break
            time.sleep(2) # サイクル間の待機
        self.stop()
        print("🧬 Awareness loop finished.")
