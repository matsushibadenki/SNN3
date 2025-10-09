# matsushibadenki/snn3/snn-cli.py
#
# 統合CLIツール (typer版)
#
# 改善点:
# - ROADMAPフェーズ8に基づき、`emergent-system`サブコマンドを追加。
# - マルチエージェントによる協調的なタスク解決プロセスを起動できるようにした。

import sys
from pathlib import Path
import asyncio
import torch
import typer
from typing import List, Optional
import random

# --- プロジェクト内モジュールのインポート ---
sys.path.append(str(Path(__file__).resolve().parent))

from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.agent.digital_life_form import DigitalLifeForm
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.rl_env.simple_env import SimpleEnvironment
import train as gradient_based_trainer
from snn_research.distillation.model_registry import SimpleModelRegistry
from snn_research.agent.memory import Memory
from snn_research.tools.web_crawler import WebCrawler
from snn_research.cognitive_architecture.rag_snn import RAGSystem
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from snn_research.cognitive_architecture.emergent_system import EmergentCognitiveSystem
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
import app.main as gradio_app
import app.langchain_main as langchain_gradio_app

# --- CLIアプリケーションの定義 ---
app = typer.Typer(
    help="Project SNN: 統合CLIツール",
    rich_markup_mode="markdown",
    add_completion=False
)

# --- サブコマンドグループの作成 ---
agent_app = typer.Typer(help="自律エージェントを操作して単一タスクを実行")
app.add_typer(agent_app, name="agent")

planner_app = typer.Typer(help="高次認知プランナーを操作して複雑なタスクを実行")
app.add_typer(planner_app, name="planner")

life_form_app = typer.Typer(help="デジタル生命体の自律ループを開始")
app.add_typer(life_form_app, name="life-form")

evolve_app = typer.Typer(help="自己進化サイクルを実行")
app.add_typer(evolve_app, name="evolve")

rl_app = typer.Typer(help="生物学的強化学習を実行")
app.add_typer(rl_app, name="rl")

ui_app = typer.Typer(help="Gradioベースの対話UIを起動")
app.add_typer(ui_app, name="ui")

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
emergent_app = typer.Typer(help="創発的なマルチエージェントシステムを操作")
app.add_typer(emergent_app, name="emergent-system")
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

# ...(既存のコマンド実装は省略)...

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
@emergent_app.command("execute", help="高レベルの目標を与え、マルチエージェントシステムに協調的に解決させます。")
def emergent_execute(
    goal: str = typer.Option(..., help="システムに達成させたい高レベルの目標")
):
    """
    EmergentCognitiveSystemを初期化し、協調的タスク解決を実行する。
    """
    print(f"🚀 Emergent System Activated. Goal: {goal}")

    # --- 依存関係の構築 (ダミー) ---
    model_registry = SimpleModelRegistry()
    rag_system = RAGSystem()
    memory = Memory()
    web_crawler = WebCrawler()
    
    # グローバルワークスペースとプランナー
    global_workspace = GlobalWorkspace(model_registry=model_registry)
    planner = HierarchicalPlanner(model_registry=model_registry, rag_system=rag_system)

    # 複数のエージェントをインスタンス化
    agent1 = AutonomousAgent(name="AutonomousAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    agent2 = AutonomousAgent(name="SpecialistAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    
    # 創発システムを構築
    emergent_system = EmergentCognitiveSystem(
        planner=planner,
        agents=[agent1, agent2],
        global_workspace=global_workspace,
        model_registry=model_registry
    )

    # タスク実行
    final_report = emergent_system.execute_task(goal)

    print("\n" + "="*20 + " ✅ FINAL REPORT " + "="*20)
    print(final_report)
    print("="*60)
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

if __name__ == "__main__":
    app()
