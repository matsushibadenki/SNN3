# matsushibadenki/snn3/SNN3-5b728d05237b1a32304ee6af1a9240f1ebfe55ff/snn-cli.py
# ファイルパス: matsushibadenki/snn3/snn-cli.py
# タイトル: 統合CLIツール (typer版)
# 機能説明: プロジェクトの全機能をサブコマンド形式で実行するための統一インターフェース。
#           argparseとtyperの混在によって発生していた引数解析エラーを解消するため、
#           typerに完全に移行。gradient-trainが追加の引数を正しく
#           train.pyに渡せるように修正。

import sys
from pathlib import Path
import asyncio
import torch
import typer
from typing import List, Optional

# --- プロジェクトルートをPythonパスに追加 ---
# これにより、プロジェクト内のモジュールを正しくインポートできる
sys.path.append(str(Path(__file__).resolve().parent))

# --- 各機能のコアロジックをインポート ---
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

# --- CLIアプリケーションの定義 ---
app = typer.Typer(
    help="Project SNN: 統合CLIツール",
    rich_markup_mode="md",
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

# --- agent サブコマンドの実装 ---
@agent_app.command("solve", help="指定されたタスクを解決します。専門家モデルの検索、オンデマンド学習、推論を実行します。")
def agent_solve(
    task: str = typer.Option(..., help="タスクの自然言語説明 (例: '感情分析')"),
    prompt: Optional[str] = typer.Option(None, help="推論を実行する場合の入力プロンプト"),
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    unlabeled_data: Optional[Path] = typer.Option(None, help="新規学習時に使用するデータパス", exists=True, file_okay=True, dir_okay=False),
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    force_retrain: bool = typer.Option(False, "--force-retrain", help="モデル登録簿を無視して強制的に再学習"),
    min_accuracy: float = typer.Option(0.6, help="専門家モデルを選択するための最低精度要件"),
    max_spikes: float = typer.Option(10000.0, help="専門家モデルを選択するための平均スパイク数上限")
):
    model_registry = SimpleModelRegistry()
    rag_system = RAGSystem()
    memory = Memory()
    web_crawler = WebCrawler()
    planner = HierarchicalPlanner(model_registry=model_registry, rag_system=rag_system)

    agent = AutonomousAgent(
        name="cli-agent",
        planner=planner,
        model_registry=model_registry,
        memory=memory,
        web_crawler=web_crawler,
        accuracy_threshold=min_accuracy,
        energy_budget=max_spikes
    )
    
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    selected_model_info = asyncio.run(agent.handle_task(
        task_description=task,
        unlabeled_data_path=str(unlabeled_data) if unlabeled_data else None,
        force_retrain=force_retrain
    ))
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    
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
    model_registry = SimpleModelRegistry()
    rag_system = RAGSystem()
    planner = HierarchicalPlanner(model_registry=model_registry, rag_system=rag_system)
    
    final_result = planner.execute_task(task_request=request, context=context)
    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)

# --- life-form サブコマンドの実装 ---
@life_form_app.command("start", help="意識ループを開始します。AIが自律的に思考・学習します。")
def life_form_start(cycles: int = typer.Option(5, help="実行する意識サイクルの回数")):
    life_form = DigitalLifeForm()
    life_form.awareness_loop(cycles=cycles)

# --- evolve サブコマンドの実装 ---
@evolve_app.command("run", help="自己進化サイクルを1回実行します。AIが自身の性能を評価し、アーキテクチャを改善します。")
def evolve_run(
    task_description: str = typer.Option(..., help="自己評価の起点となるタスク説明"),
    model_config: Path = typer.Option("configs/models/small.yaml", help="進化対象のモデル設定ファイル", exists=True),
    initial_accuracy: float = typer.Option(0.75, help="自己評価のための初期精度"),
    initial_spikes: float = typer.Option(1500.0, help="自己評価のための初期スパイク数")
):
    model_registry = SimpleModelRegistry()
    rag_system = RAGSystem()
    memory = Memory()
    web_crawler = WebCrawler()
    planner = HierarchicalPlanner(model_registry=model_registry, rag_system=rag_system)

    agent = SelfEvolvingAgent(
        name="evolving-agent",
        planner=planner,
        model_registry=model_registry,
        memory=memory,
        web_crawler=web_crawler,
        project_root=".",
        model_config_path=str(model_config)
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
    # このコマンド以降のすべての引数を取得
    train_args = ctx.args
    
    # train.py を実行するために sys.argv を一時的に書き換える
    original_argv = sys.argv
    # 最初の引数はスクリプト名である必要があるため、'train.py' を設定
    sys.argv = ["train.py"] + train_args
    
    try:
        gradient_based_trainer.main()
    finally:
        # 実行が終わったら sys.argv を元に戻す
        sys.argv = original_argv

if __name__ == "__main__":
    app()
