# ファイルパス: matsushibadenki/snn3/SNN3-190ede29139f560c909685675a68ccf65069201c/snn-cli.py
#
# 統合CLIツール (typer版)
#
# プロジェクトの全機能をサブコマンド形式で実行するための統一インターフェース。
# argparseとtyperの混在によって発生していた引数解析エラーを解消するため、
# typerに完全に移行。gradient-trainが追加の引数を正しく
# train.pyに渡せるように修正。
#
# 修正点:
# - evolve runコマンドに --training-config オプションを追加し、
#   SelfEvolvingAgentが学習パラメータを進化させられるようにした。
# - uiサブコマンドを追加し、標準UIとLangChain連携UIを
#   選択して起動できるようにした。
# - life-formサブコマンドに `explain-last-action` を追加し、
#   AIが自身の行動理由を説明する機能（自己言及）を呼び出せるようにした。
# - ROADMAPフェーズ8に基づき、`emergent-system`サブコマンドを追加。
# - マルチエージェントによる協調的なタスク解決プロセスを起動できるようにした。
#
# 改善点 (v2):
# - DigitalLifeFormのインスタンス化をDIコンテナ経由で行うように修正。

import sys
from pathlib import Path
import asyncio
import torch
import typer
from typing import List, Optional

# --- プロジェクトルートをPythonパスに追加 ---
sys.path.append(str(Path(__file__).resolve().parent))

# --- 各機能のコアロジックをインポート ---
from app.containers import AgentContainer, AppContainer
from snn_research.agent.digital_life_form import DigitalLifeForm
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.rl_env.simple_env import SimpleEnvironment
import train as gradient_based_trainer
from snn_research.distillation.model_registry import SimpleModelRegistry
from snn_research.agent.memory import Memory
from snn_research.tools.web_crawler import WebCrawler
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.cognitive_architecture.emergent_system import EmergentCognitiveSystem
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
import app.main as gradio_app
import app.langchain_main as langchain_gradio_app
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding

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

emergent_app = typer.Typer(help="創発的なマルチエージェントシステムを操作")
app.add_typer(emergent_app, name="emergent-system")


# --- agent サブコマンドの実装 ---
@agent_app.command("solve", help="指定されたタスクを解決します。専門家モデルの検索、オンデマンド学習、推論を実行します。")
def agent_solve(
    task: str = typer.Option(..., help="タスクの自然言語説明 (例: '感情分析')"),
    prompt: Optional[str] = typer.Option(None, help="推論を実行する場合の入力プロンプト"),
    unlabeled_data: Optional[Path] = typer.Option(None, help="新規学習時に使用するデータパス", exists=True, file_okay=True, dir_okay=False),
    force_retrain: bool = typer.Option(False, "--force-retrain", help="モデル登録簿を無視して強制的に再学習"),
    min_accuracy: float = typer.Option(0.6, help="専門家モデルを選択するための最低精度要件"),
    max_spikes: float = typer.Option(10000.0, help="専門家モデルを選択するための平均スパイク数上限")
):
    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")
    
    agent = AutonomousAgent(
        name="cli-agent",
        planner=container.hierarchical_planner(),
        model_registry=container.model_registry(),
        memory=container.memory(),
        web_crawler=container.web_crawler(),
        accuracy_threshold=min_accuracy,
        energy_budget=max_spikes
    )
    
    selected_model_info = asyncio.run(agent.handle_task(
        task_description=task,
        unlabeled_data_path=str(unlabeled_data) if unlabeled_data else None,
        force_retrain=force_retrain
    ))
    
    if selected_model_info and prompt:
        print("\n" + "="*20 + " 🧠 INFERENCE " + "="*20)
        print(f"入力プロンプト: {prompt}")
        asyncio.run(agent.run_inference(selected_model_info, prompt))
    elif not selected_model_info:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。")

# --- planner サブコマンドの実装 ---
@planner_app.command("execute", help="複雑なタスク要求を実行します。内部で計画を立案し、複数の専門家を連携させます。")
def planner_execute(
    request: str = typer.Option(..., help="タスク要求 (例: '記事を要約して感情を分析')"),
    context: str = typer.Option(..., help="処理対象のデータ")
):
    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")
    planner = container.hierarchical_planner()
    
    final_result = planner.execute_task(task_request=request, context=context)
    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)

# --- life-form サブコマンドの実装 ---
def get_life_form_instance() -> DigitalLifeForm:
    """DIコンテナを使用してDigitalLifeFormのインスタンスを生成するヘルパー関数"""
    agent_container = AgentContainer()
    agent_container.config.from_yaml("configs/base_config.yaml")
    app_container = AppContainer()
    app_container.config.from_yaml("configs/base_config.yaml")

    planner = agent_container.hierarchical_planner()
    model_registry = agent_container.model_registry()
    memory = agent_container.memory()
    web_crawler = agent_container.web_crawler()
    rag_system = agent_container.rag_system()

    autonomous_agent = AutonomousAgent(
        name="AutonomousAgent", planner=planner, model_registry=model_registry, 
        memory=memory, web_crawler=web_crawler
    )
    rl_agent = ReinforcementLearnerAgent(input_size=10, output_size=4, device="cpu")
    self_evolving_agent = SelfEvolvingAgent(
        name="SelfEvolvingAgent", planner=planner, model_registry=model_registry, 
        memory=memory, web_crawler=web_crawler
    )
    
    return DigitalLifeForm(
        autonomous_agent=autonomous_agent,
        rl_agent=rl_agent,
        self_evolving_agent=self_evolving_agent,
        motivation_system=IntrinsicMotivationSystem(),
        meta_cognitive_snn=MetaCognitiveSNN(),
        memory=memory,
        physics_evaluator=PhysicsEvaluator(),
        symbol_grounding=SymbolGrounding(rag_system),
        app_container=app_container
    )

@life_form_app.command("start", help="意識ループを開始します。AIが自律的に思考・学習します。")
def life_form_start(cycles: int = typer.Option(5, help="実行する意識サイクルの回数")):
    life_form = get_life_form_instance()
    life_form.awareness_loop(cycles=cycles)

@life_form_app.command("explain-last-action", help="AI自身に、直近の行動理由を自然言語で説明させます。")
def life_form_explain():
    print("🤔 AIに自身の行動理由を説明させます...")
    life_form = get_life_form_instance()
    explanation = life_form.explain_last_action()
    print("\n" + "="*20 + " 🤖 AIによる自己解説 " + "="*20)
    if explanation:
        print(explanation)
    else:
        print("説明の生成に失敗しました。")
    print("="*64)

# --- evolve サブコマンドの実装 ---
@evolve_app.command("run", help="自己進化サイクルを1回実行します。AIが自身の性能を評価し、アーキテクチャを改善します。")
def evolve_run(
    task_description: str = typer.Option(..., help="自己評価の起点となるタスク説明"),
    training_config: Path = typer.Option("configs/base_config.yaml", help="進化対象の基本設定ファイル", exists=True),
    model_config: Path = typer.Option("configs/models/small.yaml", help="進化対象のモデル設定ファイル", exists=True),
    initial_accuracy: float = typer.Option(0.75, help="自己評価のための初期精度"),
    initial_spikes: float = typer.Option(1500.0, help="自己評価のための初期スパイク数")
):
    container = AgentContainer()
    container.config.from_yaml(str(training_config))
    container.config.from_yaml(str(model_config))

    agent = SelfEvolvingAgent(
        name="evolving-agent",
        planner=container.hierarchical_planner(),
        model_registry=container.model_registry(),
        memory=container.memory(),
        web_crawler=container.web_crawler(),
        project_root=".",
        model_config_path=str(model_config),
        training_config_path=str(training_config)
    )
    initial_metrics = {
        "accuracy": initial_accuracy,
        "avg_spikes_per_sample": initial_spikes
    }
    agent.run_evolution_cycle(
        task_description=task_description,
        initial_metrics=initial_metrics
    )

# --- rl サブコマンドの実装 ---
@rl_app.command("run", help="強化学習ループを開始します。エージェントが試行錯誤から学習します。")
def rl_run(
    episodes: int = typer.Option(100, help="学習エピソード数"),
    pattern_size: int = typer.Option(10, help="環境のパターンサイズ")
):
    from tqdm import tqdm
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    env = SimpleEnvironment(pattern_size=pattern_size, device=device)
    agent = ReinforcementLearnerAgent(input_size=pattern_size, output_size=pattern_size, device=device)
    
    progress_bar = tqdm(range(episodes))
    total_reward = 0.0

    for episode in progress_bar:
        state = env.reset()
        action = agent.get_action(state)
        _, reward, _ = env.step(action)
        agent.learn(reward)
        total_reward += reward
        avg_reward = total_reward / (episode + 1)
        progress_bar.set_postfix({"Avg Reward": f"{avg_reward:.3f}"})
    
    print(f"\n✅ 学習完了。最終的な平均報酬: {total_reward / episodes:.4f}")


# --- ui サブコマンドの実装 ---
@ui_app.command("start", help="標準のGradio UIを起動します。")
def ui_start(
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
    model_path: Optional[str] = typer.Option(None, help="モデルのパス（設定ファイルを上書き）"),
):
    original_argv = sys.argv
    sys.argv = [
        "app/main.py",
        "--model_config", str(model_config),
    ]
    if model_path:
        sys.argv.extend(["--model_path", model_path])
    
    try:
        print("🚀 標準のGradio UIを起動します...")
        gradio_app.main()
    finally:
        sys.argv = original_argv

@ui_app.command("start-langchain", help="LangChain連携版のGradio UIを起動します。")
def ui_start_langchain(
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
    model_path: Optional[str] = typer.Option(None, help="モデルのパス（設定ファイルを上書き）"),
):
    original_argv = sys.argv
    sys.argv = [
        "app/langchain_main.py",
        "--model_config", str(model_config),
    ]
    if model_path:
        sys.argv.extend(["--model_path", model_path])

    try:
        print("🚀 LangChain連携版のGradio UIを起動します...")
        langchain_gradio_app.main()
    finally:
        sys.argv = original_argv

# --- emergent-system サブコマンドの実装 ---
@emergent_app.command("execute", help="高レベルの目標を与え、マルチエージェントシステムに協調的に解決させます。")
def emergent_execute(
    goal: str = typer.Option(..., help="システムに達成させたい高レベルの目標")
):
    print(f"🚀 Emergent System Activated. Goal: {goal}")

    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")

    planner = container.hierarchical_planner()
    model_registry = container.model_registry()
    memory = container.memory()
    web_crawler = container.web_crawler()
    
    global_workspace = GlobalWorkspace(model_registry=model_registry)

    agent1 = AutonomousAgent(name="AutonomousAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    agent2 = AutonomousAgent(name="SpecialistAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    
    emergent_system = EmergentCognitiveSystem(
        planner=planner,
        agents=[agent1, agent2],
        global_workspace=global_workspace,
        model_registry=model_registry
    )

    final_report = emergent_system.execute_task(goal)

    print("\n" + "="*20 + " ✅ FINAL REPORT " + "="*20)
    print(final_report)
    print("="*60)

# --- gradient-train サブコマンドの実装 ---
@app.command(
    "gradient-train",
    help="""
    勾配ベースでSNNモデルを手動学習します (train.pyを呼び出します)。
    このコマンドの後に、train.pyに渡したい引数をそのまま続けてください。
    
    例: `python snn-cli.py gradient-train --model_config configs/models/large.yaml --data_path data/sample_data.jsonl`
    """,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def gradient_train(ctx: typer.Context):
    print("🔧 勾配ベースの学習プロセスを開始します...")
    train_args = ctx.args
    
    original_argv = sys.argv
    sys.argv = ["train.py"] + train_args
    
    try:
        gradient_based_trainer.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    app()
