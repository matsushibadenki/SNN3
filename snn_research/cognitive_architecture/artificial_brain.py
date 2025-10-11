# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# (新規作成)
#
# Title: Artificial Brain (人工脳) オーケストレーター
#
# Description:
# - これまでに実装された認知アーキテクチャの各コンポーネントを
#   一つのシステムとして統合し、全体の情報処理フローを制御する。
# - 感覚入力から思考、意思決定、行動出力までの一連のプロセスを管理する。
# - システムの状態を保持し、各モジュール間の連携を司る。

from typing import Dict, Any, List

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
    def __init__(self):
        print("🚀 人工脳システムの起動を開始...")
        # 各認知モジュールをインスタンス化
        self.pfc = PrefrontalCortex()
        self.hippocampus = Hippocampus(capacity=50)
        self.cortex = Cortex()
        self.amygdala = Amygdala()
        self.basal_ganglia = BasalGanglia()
        self.cerebellum = Cerebellum()
        self.motor_cortex = MotorCortex(actuators=['arm', 'gripper', 'voice'])
        
        # システム全体のコンテキストを保持
        self.global_context: Dict[str, Any] = {
            "internal_state": {},
            "external_request": None
        }
        print("✅ 人工脳システムの全モジュールが正常に起動しました。")

    def process_sensory_input(self, text_input: str):
        """
        外部からの感覚入力（ここではテキスト）を受け取り、
        知覚から行動までの一連の認知プロセスを実行する。
        """
        print(f"\n--- 🧠 新しい入力情報を受信 --- \n'{text_input}'")
        
        # 1. 記憶: 入力情報を短期記憶にエピソードとして保存
        self.hippocampus.store_episode({'type': 'sensory_input', 'content': text_input})

        # 2. 情動評価: 入力に対する情動価を評価
        emotion = self.amygdala.evaluate_emotion(text_input)
        self.global_context['internal_state']['emotion'] = emotion
        print(f"💖 扁桃体による評価: {emotion}")

        # 3. 目標設定: 現在の状況に基づき、次の高レベル目標を決定
        goal = self.pfc.decide_goal(self.global_context)
        
        # (この後の実装は簡略化)
        # 4. 計画: HierarchicalPlannerが目標を具体的な行動候補に分解 (ダミー)
        action_candidates = [
            {'action': 'analyze_text', 'value': 0.8, 'duration': 1.5},
            {'action': 'search_web', 'value': 0.6, 'duration': 3.0},
            {'action': 'generate_response', 'value': 0.9, 'duration': 2.0}
        ]
        
        # 5. 行動選択: 大脳基底核が最適な行動を選択
        selected_action = self.basal_ganglia.select_action(action_candidates)

        if selected_action:
            # 6. 運動制御: 小脳が行動を精密なコマンドに変換
            motor_commands = self.cerebellum.refine_action_plan(selected_action)

            # 7. 行動実行: 運動野がコマンドを実行
            self.motor_cortex.execute_commands(motor_commands)

        print("--- ✅ 認知サイクル完了 ---")