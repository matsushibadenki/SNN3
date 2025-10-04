# snn_research/agent/digital_life_form.py
# DigitalLifeForm オーケストレーター
# 概要：内発的動機付けとメタ認知に基づき、各種エージェントを自律的に起動するマスタープロセス。
import time
import logging
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
# 各エージェントのインポート（実際のパスに合わせて修正が必要）
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import RLAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.planner_snn import PlannerSNN


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
        
        # 各種エージェントのインスタンス化
        self.autonomous_agent = AutonomousAgent()
        self.rl_agent = RLAgent()
        self.self_evolving_agent = SelfEvolvingAgent()
        self.planner_agent = PlannerSNN() # PlannerSNNをエージェントとして利用
        
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
            # (execute_actionの結果から得られる実際の値で更新する)
            # 以下はダミーの更新
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
        
        # 優先順位1: 緊急性の高い問題への対応 (メタ認知評価に基づく)
        if performance_eval["status"] == "knowledge_gap":
            logging.info("Reason: Knowledge gap detected. Acquiring new information.")
            return "acquire_new_knowledge" # 自律エージェントによるWeb学習
        
        if performance_eval["status"] == "capability_gap":
            logging.info("Reason: Capability gap detected. Evolving model architecture.")
            return "evolve_architecture" # 自己進化エージェント

        # 優先順位2: 内発的動機付けに基づく行動
        if internal_state["boredom"] > 0.7 and internal_state["confidence"] > 0.8:
            logging.info("Reason: High boredom and confidence. Exploring new tasks.")
            return "explore_new_task_with_rl" # 強化学習エージェントで新タスク探索
            
        if internal_state["curiosity"] > 0.6:
            logging.info("Reason: High curiosity. Planning complex task.")
            return "plan_and_execute" # プランナーで複雑なタスクを実行

        # デフォルト行動
        logging.info("Reason: Default behavior. Practicing existing skills.")
        return "practice_skill_with_rl" # 強化学習エージェントで既存スキルを練習

    def _execute_action(self, action):
        """
        指定されたアクションに対応するエージェントを実行する。
        
        Returns:
            tuple: (result_dict, reward, expert_used_list)
        """
        try:
            if action == "acquire_new_knowledge":
                # result = self.autonomous_agent.search_and_learn("latest SNN research trends")
                return {"status": "success", "info": "Web crawling completed.", "accuracy": 0.96}, 0.8, ["web_crawler"]
            elif action == "evolve_architecture":
                # result = self.self_evolving_agent.evolve()
                return {"status": "success", "info": "Evolution completed.", "accuracy": 0.97}, 0.9, ["self_evolver"]
            elif action == "explore_new_task_with_rl":
                # result, reward = self.rl_agent.run_episode(explore=True)
                return {"status": "success", "info": "Exploration finished.", "accuracy": 0.92}, 0.7, ["rl_agent_explorer"]
            elif action == "practice_skill_with_rl":
                # result, reward = self.rl_agent.run_episode(explore=False)
                return {"status": "success", "info": "Practice finished.", "accuracy": 0.98}, 0.5, ["rl_agent_practicer"]
            elif action == "plan_and_execute":
                # plan = self.planner_agent.create_plan("Summarize text and analyze sentiment")
                # result = self.planner_agent.execute_plan(plan)
                return {"status": "success", "info": "Plan executed.", "accuracy": 0.95}, 0.8, ["planner", "summarizer_snn", "sentiment_snn"]
            else:
                return {"status": "failed", "info": "Unknown action"}, 0.0, []
        except Exception as e:
            logging.error(f"Error executing action '{action}': {e}")
            return {"status": "error", "info": str(e)}, -1.0, []
