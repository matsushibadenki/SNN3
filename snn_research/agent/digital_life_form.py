# ファイルパス: matsushibadenki/snn3/SNN3-190ede29139f560c909685675a68ccf65069201c/snn_research/agent/digital_life_form.py
#
# DigitalLifeForm オーケストレーター
#
# 概要：内発的動機付けとメタ認知に基づき、各種エージェントを自律的に起動するマスタープロセス。
#
# 改善点:
# - ROADMAPフェーズ7「自己言及」を実装。
# - LangChain連携機能を利用し、自身の行動ログを基に、
#   「なぜその行動を取ったのか」を自然言語で説明する`explain_last_action`メソッドを追加。
# - ROADMAPフェーズ7「記号創発」を実装。
# - SymbolGroundingモジュールを統合し、未知の経験から新しい概念を自律的に生成する機能を追加。
#
# 改善点 (v2):
# - 依存性注入に対応するため、__init__メソッドで必要なコンポーネントを受け取るように変更。
#
# 改善点 (v3):
# - self.state の型ヒントを明示し、mypyエラーを修正。

import time
import logging
import torch
import random
import json
from typing import Dict, Any, Optional

from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from app.containers import AppContainer
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitalLifeForm:
    """
    内発的動機付けシステムとメタ認知SNNを統合し、
    永続的で自己駆動する学習ループを実現するオーケストレーター。
    """
    def __init__(
        self,
        autonomous_agent: AutonomousAgent,
        rl_agent: ReinforcementLearnerAgent,
        self_evolving_agent: SelfEvolvingAgent,
        motivation_system: IntrinsicMotivationSystem,
        meta_cognitive_snn: MetaCognitiveSNN,
        memory: Memory,
        physics_evaluator: PhysicsEvaluator,
        symbol_grounding: SymbolGrounding,
        app_container: AppContainer
    ):
        self.autonomous_agent = autonomous_agent
        self.rl_agent = rl_agent
        self.self_evolving_agent = self_evolving_agent
        self.motivation_system = motivation_system
        self.meta_cognitive_snn = meta_cognitive_snn
        self.memory = memory
        self.physics_evaluator = physics_evaluator
        self.symbol_grounding = symbol_grounding
        self.app_container = app_container
        
        self.running = False
        # mypyエラー修正: 型ヒントを明示的に指定
        self.state: Dict[str, Any] = {"last_action": None, "last_result": None}


    def start(self):
        self.running = True
        logging.info("DigitalLifeForm activated. Starting autonomous loop.")
        self.life_cycle()

    def stop(self):
        self.running = False
        logging.info("DigitalLifeForm deactivating.")

    def life_cycle(self):
        while self.running:
            internal_state = self.motivation_system.get_internal_state()
            performance_eval = self.meta_cognitive_snn.evaluate_performance()
            dummy_mem_sequence = torch.randn(100)
            dummy_spikes = (torch.rand(100) > 0.8).float()
            physical_rewards = self.physics_evaluator.evaluate_physical_consistency(dummy_mem_sequence, dummy_spikes)
            
            action = self._decide_next_action(internal_state, performance_eval, physical_rewards)
            
            result, external_reward, expert_used = self._execute_action(action)

            self.symbol_grounding.process_observation(result, context=f"action '{action}'")
            
            reward_vector = {
                "external": external_reward,
                "physical": physical_rewards
            }
            decision_context = {"internal_state": internal_state, "performance_eval": performance_eval, "physical_rewards": physical_rewards}
            self.memory.record_experience(self.state, action, result, reward_vector, expert_used, decision_context)
            
            # 以下はダミーのメトリクス更新
            dummy_prediction_error = result.get("prediction_error", 0.1) if isinstance(result, dict) else 0.1
            dummy_success_rate = result.get("success_rate", 0.9) if isinstance(result, dict) else 0.9
            dummy_task_similarity = 0.8
            dummy_loss = result.get("loss", 0.05) if isinstance(result, dict) else 0.05
            dummy_time = result.get("computation_time", 1.0) if isinstance(result, dict) else 1.0
            dummy_accuracy = result.get("accuracy", 0.95) if isinstance(result, dict) else 0.95

            self.motivation_system.update_metrics(dummy_prediction_error, dummy_success_rate, dummy_task_similarity, dummy_loss)
            self.meta_cognitive_snn.update_metadata(dummy_loss, dummy_time, dummy_accuracy)
            self.state = {"last_action": action, "last_result": result}
            
            logging.info(f"Action: {action}, Result: {result}, Reward: {external_reward}")
            logging.info(f"New Internal State: {self.motivation_system.get_internal_state()}")
            
            time.sleep(10)

    def _decide_next_action(self, internal_state: Dict[str, float], performance_eval: Dict[str, Any], physical_rewards: Dict[str, float]) -> str:
        action_scores: Dict[str, float] = {
            "acquire_new_knowledge": 0.0,
            "evolve_architecture": 0.0,
            "explore_new_task_with_rl": 0.0,
            "plan_and_execute": 0.0,
            "practice_skill_with_rl": 0.0,
        }

        if performance_eval.get("status") == "knowledge_gap":
            action_scores["acquire_new_knowledge"] += 10.0
            logging.info("Decision reason: Knowledge gap detected.")
        
        if performance_eval.get("status") == "capability_gap":
            action_scores["evolve_architecture"] += 5.0
            logging.info("Decision reason: Capability gap detected.")
        if physical_rewards.get("sparsity_reward", 1.0) < 0.5:
            action_scores["evolve_architecture"] += 8.0
            logging.info("Decision reason: Low energy efficiency (sparsity).")

        action_scores["explore_new_task_with_rl"] += internal_state.get("curiosity", 0.5) * 5.0
        action_scores["plan_and_execute"] += internal_state.get("curiosity", 0.5) * 3.0
        
        if internal_state.get("boredom", 0.0) > 0.7:
            action_scores["explore_new_task_with_rl"] += internal_state.get("boredom", 0.0) * 10.0
            logging.info("Decision reason: High boredom.")

        action_scores["practice_skill_with_rl"] += internal_state.get("confidence", 0.5) * 2.0
        action_scores["explore_new_task_with_rl"] += internal_state.get("confidence", 0.5) * 1.0

        action_scores["practice_skill_with_rl"] += 1.0

        actions = list(action_scores.keys())
        scores = [max(0, s) for s in action_scores.values()]
        total_score = sum(scores)

        if total_score == 0:
            chosen_action = "practice_skill_with_rl"
        else:
            probabilities = [s / total_score for s in scores]
            chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        
        logging.info(f"Action scores: {action_scores}")
        if total_score > 0:
            logging.info(f"Probabilities: { {a: f'{p:.2%}' for a, p in zip(actions, probabilities)} }")
        logging.info(f"Chosen action: {chosen_action}")

        return chosen_action

    def _execute_action(self, action: str) -> tuple[Dict[str, Any], float, List[str]]:
        try:
            if action == "acquire_new_knowledge":
                result_str = self.autonomous_agent.learn_from_web("latest SNN research trends")
                return {"status": "success", "info": result_str, "accuracy": 0.96}, 0.8, ["web_crawler"]
            elif action == "evolve_architecture":
                result_str = self.self_evolving_agent.evolve()
                return {"status": "success", "info": result_str, "accuracy": 0.97}, 0.9, ["self_evolver"]
            elif action == "explore_new_task_with_rl":
                return {"status": "success", "info": "Exploration finished.", "accuracy": 0.92}, 0.7, ["rl_agent_explorer"]
            elif action == "practice_skill_with_rl":
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
        print(f"🧬 Digital Life Form awareness loop starting for {cycles} cycles.")
        self.running = True
        for i in range(cycles):
            if not self.running:
                break
            print(f"\n----- Cycle {i+1}/{cycles} -----")
            self.life_cycle_step() # 1サイクル分の処理を呼び出し
            time.sleep(2)
        self.stop()
        print("🧬 Awareness loop finished.")
    
    def life_cycle_step(self):
        """life_cycleの1回分の処理"""
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
        
        # メトリクス更新
        if isinstance(result, dict):
            self.motivation_system.update_metrics(result.get("prediction_error", 0.1), result.get("success_rate", 0.9), 0.8, result.get("loss", 0.05))
            self.meta_cognitive_snn.update_metadata(result.get("loss", 0.05), result.get("computation_time", 1.0), result.get("accuracy", 0.95))

        self.state = {"last_action": action, "last_result": result}
        logging.info(f"Action: {action}, Result: {result}, Reward: {reward}")

    def explain_last_action(self) -> Optional[str]:
        """
        直近の行動ログを基に、自身の行動理由を自然言語で説明する。
        """
        try:
            with open(self.memory.memory_path, "rb") as f:
                try:
                    f.seek(-2, 2)
                    while f.read(1) != b'\n':
                        f.seek(-2, 1)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
            
            last_experience = json.loads(last_line)
        except (IOError, json.JSONDecodeError, IndexError):
            return "行動履歴が見つかりません。"

        prompt = f"""
        あなたは、自身の行動を分析し、その理由を分かりやすく説明するAIです。
        以下の内部ログは、あなた自身の直近の行動記録です。この記録を基に、なぜその行動を取ったのかを一人称（「私」）で説明してください。

        ### 行動ログ
        - **実行した行動:** {last_experience.get('action')}
        - **意思決定の根拠:**
          - **内発的動機（内部状態）:** {last_experience.get('decision_context', {}).get('internal_state')}
          - **自己パフォーマンス評価:** {last_experience.get('decision_context', {}).get('performance_eval')}
          - **物理効率評価:** {last_experience.get('decision_context', {}).get('physical_rewards')}

        ### 指示
        上記の根拠を統合し、あなたの思考プロセスを平易な言葉で説明してください。
        """
        print("\n--- 自己言及プロンプト ---")
        print(prompt)
        print("--------------------------\n")

        try:
            snn_llm = self.app_container.langchain_adapter()
            explanation = snn_llm._call(prompt)
            return explanation
        except Exception as e:
            logging.error(f"LLMによる自己言及の生成に失敗しました: {e}")
            return "エラー: 自己言及の生成に失敗しました。"
