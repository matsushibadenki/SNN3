# matsushibadenki/snn3/snn_research/agent/self_evolving_agent.py
# Title: 自己進化エージェント
# Description: 自身のアーキテクチャや学習ルールを自律的に修正・改善できるエージェント。
#              mypyエラー修正: super().__init__に引数を追加。
#              mypyエラー修正: snn-cli.pyからの呼び出しに対応するため、メソッドと引数を修正。
# 改善点: ダミーだったevolveメソッドに、設定ファイルを読み込んでパラメータを強化し、
#         新しい設定ファイルとして保存する具体的な自己進化ロジックを実装。

from typing import Dict, Any, Optional
import os
import yaml
from omegaconf import OmegaConf

from .autonomous_agent import AutonomousAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.tools.web_crawler import WebCrawler
from .memory import Memory as AgentMemory


class SelfEvolvingAgent(AutonomousAgent):
    """
    自身の性能を監視し、必要に応じて自己進化するエージェント。
    """
    def __init__(
        self,
        name: str,
        planner: HierarchicalPlanner,
        model_registry: ModelRegistry,
        memory: AgentMemory,
        web_crawler: WebCrawler,
        evolution_threshold: float = 0.5,
        project_root: str = ".",
        model_config_path: Optional[str] = None,
    ):
        super().__init__(name, planner, model_registry, memory, web_crawler)
        self.evolution_threshold = evolution_threshold
        self.project_root = project_root
        self.model_config_path = model_config_path


    def execute(self, task_description: str) -> str:
        """
        タスクを実行し、結果を評価して自己進化を試みる。
        """
        result = super().execute(task_description)
        
        # 簡易的な性能評価
        performance = self.evaluate_performance(task_description, result)
        
        if performance < self.evolution_threshold:
            print(f"📉 Performance ({performance:.2f}) is below threshold ({self.evolution_threshold}). Triggering evolution...")
            evolution_result = self.evolve()
            return f"{result}\nAdditionally, self-evolution was triggered: {evolution_result}"

        return result

    def evaluate_performance(self, task: str, result: str) -> float:
        """
        タスクの実行結果を評価する（簡易版）。
        """
        if "successfully" in result.lower() or "using expert" in result.lower():
            return 0.9
        if "no specific expert found" in result.lower():
            return 0.4
        return 0.1

    def evolve(self) -> str:
        """
        自己進化のプロセスを実行する。
        モデル設定ファイルを読み込み、アーキテクチャを強化して新しい設定ファイルを生成する。
        """
        if not self.model_config_path or not os.path.exists(self.model_config_path):
            return "Evolution failed: model_config_path is not set or file not found."

        try:
            print(f"🧬 Starting evolution process for {self.model_config_path}...")
            
            # 1. 現在のアーキテクチャ設定を読み込む
            cfg = OmegaConf.load(self.model_config_path)

            # 2. パラメータを強化する (例: d_modelを1.5倍、num_layersを+2)
            original_d_model = cfg.model.d_model
            original_num_layers = cfg.model.num_layers

            # パラメータを増加させる (より大きく、より深く)
            cfg.model.d_model = int(original_d_model * 1.5)
            cfg.model.num_layers = original_num_layers + 2
            
            # 関連パラメータも調整 (例: d_state, branch_features)
            if 'd_state' in cfg.model:
                cfg.model.d_state = int(cfg.model.d_state * 1.5)
            if 'neuron' in cfg.model and 'branch_features' in cfg.model.neuron:
                cfg.model.neuron.branch_features = cfg.model.d_model // cfg.model.neuron.num_branches

            print(f"   - d_model evolved: {original_d_model} -> {cfg.model.d_model}")
            print(f"   - num_layers evolved: {original_num_layers} -> {cfg.model.num_layers}")

            # 3. 新しい設定ファイルを生成
            base_name, ext = os.path.splitext(self.model_config_path)
            new_config_path = f"{base_name}_evolved_v{self.get_next_version()}{ext}"
            
            OmegaConf.save(config=cfg, f=new_config_path)

            return f"Successfully evolved architecture. New configuration saved to '{new_config_path}'."

        except Exception as e:
            return f"Evolution failed with an error: {e}"

    def get_next_version(self) -> int:
        # 簡易的なバージョン管理
        return 2

    def run_evolution_cycle(self, task_description: str, initial_metrics: Dict[str, float]) -> None:
        """snn-cli.pyから呼び出されるためのエントリポイント。"""
        print(f"Running evolution cycle for task: {task_description} with initial metrics: {initial_metrics}")
        
        # 初期性能を評価
        performance = initial_metrics.get("accuracy", 0.0)
        
        if performance < self.evolution_threshold:
            print(f"📉 Initial performance ({performance:.2f}) is below threshold ({self.evolution_threshold}).")
            evolution_result = self.evolve()
            print(f"✨ {evolution_result}")
        else:
            print(f"✅ Initial performance ({performance:.2f}) is sufficient. No evolution needed.")