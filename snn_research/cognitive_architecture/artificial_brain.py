# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# (更新)
# 改善点: 依存性注入(DI)パターンを採用し、各モジュールをコンストラクタで受け取るように変更。
# 改善点: process_sensory_inputメソッドを完全に実装し、感覚入力から行動出力までの
#          一連の認知プロセスをシミュレートする。

from typing import Dict, Any, List

from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from .perception_cortex import PerceptionCortex
from .prefrontal_cortex import PrefrontalCortex
from .hippocampus import Hippocampus
from .cortex import Cortex
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex

class ArtificialBrain:
    """
    認知アーキテクチャ全体を統合・制御する人工脳システム。
    """
    def __init__(
        self,
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        perception_cortex: PerceptionCortex,
        hippocampus: Hippocampus,
        cortex: Cortex,
        amygdala: Amygdala,
        prefrontal_cortex: PrefrontalCortex,
        basal_ganglia: BasalGanglia,
        cerebellum: Cerebellum,
        motor_cortex: MotorCortex,
        actuator: Actuator
    ):
        print("🚀 人工脳システムの起動を開始...")
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.perception = perception_cortex
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.pfc = prefrontal_cortex
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.motor = motor_cortex
        self.actuator = actuator
        
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
        #    (短期記憶や情動状態をコンテキストとして渡す)
        self.global_context['recent_memory'] = self.hippocampus.retrieve_recent_episodes(1)
        goal = self.pfc.decide_goal(self.global_context)
        
        # 7. 計画: HierarchicalPlannerが目標を具体的な行動候補に分解 (ダミー)
        #    (長期的にはプランナーもDIで受け取る)
        action_candidates = self._generate_action_candidates(goal, perception_result)
        
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

    def _generate_action_candidates(self, goal: str, perception: Dict[str, Any]) -> List[Dict[str, Any]]:
        """現在の目標と知覚に基づき、行動の選択肢を生成する（ダミー実装）。"""
        # ここでは簡易的なルールベースで行動候補を生成
        if "analyze" in goal.lower():
            return [{'action': 'analyze_features', 'value': 0.9, 'duration': 1.0}]
        if "response" in goal.lower():
            return [{'action': 'generate_voice_response', 'value': 0.85, 'duration': 2.5}]
        return [{'action': 'observe', 'value': 0.5, 'duration': 0.5}]
