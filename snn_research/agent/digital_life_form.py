# matsushibadenki/snn2/snn_research/agent/digital_life_form.py
# Phase 6: 全ての認知機能を統合し、自律的に活動するデジタル生命体
# 変更点:
# - PhysicsEvaluatorと進化したIntrinsicMotivationSystemを統合。
# - 意識ループ内に「内省ステップ」を追加し、物理状態（膜電位など）を観測。
# - 予測誤差だけでなく、物理法則の一致度も考慮して次の行動を決定するように進化。
# - mypyエラー(attr-defined)を修正するため、__init__にself.device属性を追加。

import time
import torch
from typing import Dict, Any

from .self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.emergent_system import EmergentSystem
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator

class DigitalLifeForm(SelfEvolvingAgent):
    """
    自己の内部状態（好奇心や矛盾）に基づき、永続的に活動する
    デジタル生命体のコアロジック。
    """
    def __init__(self, project_root: str = "."):
        super().__init__(project_root)
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        # 物理法則評価器と、それを利用する新しい動機付けシステムを初期化
        self.physics_evaluator = PhysicsEvaluator()
        self.motivation_system = IntrinsicMotivationSystem(
            physics_evaluator=self.physics_evaluator,
            window_size=10
        )
        
        self.emergent_system = EmergentSystem()
        self.planner = HierarchicalPlanner()
        
        # 初期状態として、ダミーの性能指標とエラーを持つ
        self.current_metrics: Dict[str, Any] = {"accuracy": 0.8, "avg_spikes_per_sample": 1200.0}
        self.last_prediction_error = 0.1

    def _introspect(self) -> Dict[str, torch.Tensor]:
        """
        自己の内部状態を観測するための内省ステップ。
        ダミーの推論を実行し、物理的な状態（膜電位、スパイク）を取得する。
        """
        # プランナーSNNが思考の代行を行う (ロードされていなければ何もしない)
        if not self.planner.planner_snn:
            return {
                "mem_sequence": torch.zeros(1),
                "spikes": torch.zeros(1),
            }
        
        with torch.no_grad():
            dummy_input = torch.randint(0, 1000, (1, self.planner.planner_snn.time_steps), device=self.device)
            # 物理状態を取得するために、return_full_mems=True, return_spikes=True で実行
            _, spikes, mem_sequence = self.planner.planner_snn(
                dummy_input,
                return_spikes=True,
                return_full_mems=True
            )
        return {
            "mem_sequence": mem_sequence,
            "spikes": spikes,
        }

    def awareness_loop(self, cycles: int = 5):
        """
        デジタル生命体の意識と活動のコアとなる自律ループ。
        """
        print("\n" + "="*20 + "🧬 デジタル生命体 意識ループ開始 🧬" + "="*20)
        
        for i in range(cycles):
            print(f"\n--- [意識サイクル {i+1}/{cycles}] ---")

            # 1. 内省 (Introspection)
            #    自己の内部モデルの物理的な状態を観測する
            internal_state = self._introspect()
            
            # 2. 動機付けの更新 (Motivation Update)
            #    予測誤差と物理的整合性の両方から、現在の「やる気」を決定する
            self.motivation_system.update_motivation(
                current_error=self.last_prediction_error,
                mem_sequence=internal_state["mem_sequence"],
                spikes=internal_state["spikes"],
                physics_reward_weight=0.2 # 物理報酬の重み
            )
            
            # 3. 行動決定 (Action Selection)
            #    「やる気」レベルに基づき、探求するか、自己改善するかを決定する
            if self.motivation_system.should_explore():
                print("🤔 好奇心に基づき、新たな探求を開始します...")
                
                # 3a. 探求 (Exploration)
                synthesis_result = self.emergent_system.synthesize_responses(
                    prompt="現在、最も意見が分かれている概念は何か？", 
                    domain="文章要約" # 例としてドメインを指定
                )
                print(f"🔍 探求テーマ: {synthesis_result}")
                
                task_request = f"'{synthesis_result}' について、関連情報を調査し、新たな知見を要約せよ。"
                plan_result = self.planner.execute_task(task_request, context=synthesis_result)
                print(f"💡 探求結果: {plan_result}")
                
                # (シミュレーション) 探求によって世界の理解が少し進み、次の予測誤差が少し減る
                self.last_prediction_error *= 0.95
            
            else:
                print("😌 現在の理解に満足しています。自己の性能改善を試みます。")
                # 3b. 自己改善 (Self-Improvement / Exploitation)
                if self.current_metrics["accuracy"] < 0.95:
                     self.run_evolution_cycle("汎用言語理解", self.current_metrics)
                     # (シミュレーション) 自己進化により性能が向上
                     self.current_metrics["accuracy"] *= 1.05
                else:
                    print("✅ 自己性能にも満足しています。安定状態を維持します。")

            time.sleep(1) # サイクル間の小休止

        print("\n" + "="*20 + "🧬 デジタル生命体 意識ループ終了 🧬" + "="*20)

