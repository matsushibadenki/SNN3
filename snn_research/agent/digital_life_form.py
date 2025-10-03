# matsushibadenki/snn2/snn_research/agent/digital_life_form.py
# Phase 6: 全ての認知機能を統合し、自律的に活動するデジタル生命体
# 変更点:
# - PhysicsEvaluatorと進化したIntrinsicMotivationSystemを統合。
# - 意識ループ内に「内省ステップ」を追加し、物理状態（膜電位など）を観測。
# - 予測誤差だけでなく、物理法則の一致度も考慮して次の行動を決定するように進化。
# - mypyエラー(attr-defined)を修正するため、__init__にself.device属性を追加。
# - ◾️◾️◾️↓修正開始◾️◾️◾️
# - WebCrawlerToolとKnowledgeDistillationManagerを導入し、アイドル時に自律的にWebから学習する機能を追加。

import time
import torch
from typing import Dict, Any

from .self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.emergent_system import EmergentSystem
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from snn_research.tools.web_crawler import WebCrawler
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️


class DigitalLifeForm(SelfEvolvingAgent):
# ... existing code ...
        self.physics_evaluator = PhysicsEvaluator()
        self.motivation_system = IntrinsicMotivationSystem(
            physics_evaluator=self.physics_evaluator,
            window_size=10
        )
        
        self.emergent_system = EmergentSystem()
        self.planner = HierarchicalPlanner()
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 自律学習のためのコンポーネント
        self.web_crawler = WebCrawler()
        self.distillation_manager = KnowledgeDistillationManager(
            base_config_path="configs/base_config.yaml",
            model_config_path="configs/models/small.yaml"
        )
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        # 初期状態として、ダミーの性能指標とエラーを持つ
        self.current_metrics: Dict[str, Any] = {"accuracy": 0.8, "avg_spikes_per_sample": 1200.0}
        self.last_prediction_error = 0.1
# ... existing code ...
            # 3. 行動決定 (Action Selection)
            #    「やる気」レベルに基づき、探求するか、自己改善するかを決定する
            if self.motivation_system.should_explore():
                print("🤔 好奇心に基づき、新たな探求を開始します...")
                
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                # 3a. 探求 (Exploration) - Webからの自律学習を追加
                self._explore_the_web()
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            
            else:
                print("😌 現在の理解に満足しています。自己の性能改善を試みます。")
# ... existing code ...
                     self.current_metrics["accuracy"] *= 1.05
                else:
                    print("✅ 自己性能にも満足しています。安定状態を維持します。")

            time.sleep(1) # サイクル間の小休止

        print("\n" + "="*20 + "🧬 デジタル生命体 意識ループ終了 🧬" + "="*20)
        
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _explore_the_web(self):
        """
        Webを巡回し、新しいトピックについて自律的に学習する探求行動。
        """
        print("🌍 Webを巡回し、新しい知識の獲得を試みます...")
        
        # ここでは例として固定のURLとトピックを使用
        # 将来的には、自身の知識の欠損部分から探求テーマを決定する
        start_url = "https://www.itmedia.co.jp/news/subtop/aiplus/"
        topic = "最新のAI技術"
        
        # 1. データ収集
        crawled_data_path = self.web_crawler.crawl(start_url=start_url, max_pages=5)
        
        # 2. オンデマンド学習
        if crawled_data_path:
            print(f"📚 収集したデータを用いて、トピック「{topic}」に関する新しい専門家を育成します...")
            self.distillation_manager.run_on_demand_pipeline(
                task_description=topic,
                unlabeled_data_path=crawled_data_path,
                teacher_model_name="gpt2",
                force_retrain=True
            )
            # (シミュレーション) 新しい知識を得たことで、次の予測誤差が少し減る
            self.last_prediction_error *= 0.9
        else:
            print("🕸️ Webから有効なデータを収集できませんでした。")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
