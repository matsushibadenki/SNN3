# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# (更新)
# 改善点: DIパターンを拡張し、HierarchicalPlannerもコンストラクタで受け取るように変更。
# 改善点: run_cognitive_cycleを完全に実装し、感覚入力から行動出力までの
#          一連の認知プロセスをシミュレートする。

from typing import Dict, Any, List
import asyncio

# IO and encoding
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
# Core cognitive modules
from .perception_cortex import PerceptionCortex
from .prefrontal_cortex import PrefrontalCortex
from .hierarchical_planner import HierarchicalPlanner
# Memory systems
from .hippocampus import Hippocampus
from .cortex import Cortex
# Value and action selection
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
# Motor control
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex

class ArtificialBrain:
    """
    認知アーキテクチャ全体を統合・制御する人工脳システム。
    """
    def __init__(
        self,
        # Input/Output
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        actuator: Actuator,
        # Core Cognitive Flow
        perception_cortex: PerceptionCortex,
        prefrontal_cortex: PrefrontalCortex,
        hierarchical_planner: HierarchicalPlanner,
        # Memory
        hippocampus: Hippocampus,
        cortex: Cortex,
        # Value and Action
        amygdala: Amygdala,
        basal_ganglia: BasalGanglia,
        # Motor
        cerebellum: Cerebellum,
        motor_cortex: MotorCortex
    ):
        print("🚀 人工脳システムの起動を開始...")
        # I/O Modules
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        # Cognitive Modules
        self.perception = perception_cortex
        self.pfc = prefrontal_cortex
        self.planner = hierarchical_planner
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.motor = motor_cortex
        
        self.global_context: Dict[str, Any] = {
            "internal_state": {}, "external_request": None
        }
        print("✅ 人工脳システムの全モジュールが正常に起動しました。")

    def run_cognitive_cycle(self, raw_input: Any):
        """
        外部からの感覚入力（テキストなど）を受け取り、
        知覚から行動までの一連の認知プロセスを実行する。
        """
        print(f"\n--- 🧠 新しい認知サイクルを開始 --- \n入力: '{raw_input}'")
        
        # 1. 入力層: 感覚情報を受信
        sensory_info = self.receptor.receive(raw_input)

        # 2. 符号化層: 感覚情報をスパイクパターンに変換
        spike_pattern = self.encoder.encode(sensory_info, duration=50)

        # 3. 知覚層: スパイクパターンから特徴を抽出
        perception_result = self.perception.perceive(spike_pattern)
        
        # 4. 記憶（短期）: 知覚結果を短期記憶にエピソードとして保存
        episode = {'type': 'perception', 'content': perception_result, 'source_input': raw_input}
        self.hippocampus.store_episode(episode)

        # 5. 情動評価: 入力テキストに対する情動価を評価
        emotion = self.amygdala.evaluate_emotion(raw_input if isinstance(raw_input, str) else "")
        self.global_context['internal_state']['emotion'] = emotion
        print(f"💖 扁桃体による評価: {emotion}")

        # 6. 目標設定: 現在の状況に基づき、次の高レベル目標を決定
        self.global_context['recent_memory'] = self.hippocampus.retrieve_recent_episodes(1)
        goal = self.pfc.decide_goal(self.global_context)
        
        # 7. 計画: HierarchicalPlannerが目標を具体的な行動候補に分解
        plan = asyncio.run(self.planner.create_plan(goal))
        action_candidates = self._convert_plan_to_candidates(plan)
        
        # 8. 行動選択: 大脳基底核が最適な行動を選択
        selected_action = self.basal_ganglia.select_action(action_candidates)

        if selected_action:
            # 9. 運動制御: 小脳が行動を精密なコマンドに変換
            motor_commands = self.cerebellum.refine_action_plan(selected_action)

            # 10. 行動実行: 運動野がコマンドを実行
            command_logs = self.motor.execute_commands(motor_commands)

            # 11. 出力層: アクチュエータが最終的なアクションを実行
            self.actuator.run_command_sequence(command_logs)

        print("--- ✅ 認知サイクル完了 ---")

    def _convert_plan_to_candidates(self, plan) -> List[Dict[str, Any]]:
        """プランナーからの計画を、大脳基底核が解釈できる行動候補リストに変換する。"""
        candidates = []
        for task in plan.task_list:
            # ここでは単純に価値を固定値とするが、将来的には予測モデルで計算
            candidates.append({
                'action': task.get('task', 'unknown_action'),
                'value': 0.8, # 計画されたタスクは価値が高いと仮定
                'duration': 1.0 # デフォルト持続時間
            })
        return candidates
