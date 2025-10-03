# /snn-cli.py
# Title: 統合CLIツール
# Description: プロジェクトの全機能をサブコマンド形式で実行するための統一インターフェース。

import argparse
import sys
import os
from pathlib import Path

# --- プロジェクトルートをPythonパスに追加 ---
sys.path.append(str(Path(__file__).resolve().parent))


# --- 各機能のコアロジックをインポート ---
from snn_research.agent import AutonomousAgent, SelfEvolvingAgent, DigitalLifeForm, ReinforcementLearnerAgent
from snn_research.cognitive_architecture import HierarchicalPlanner

from snn_research.rl_env.simple_env import SimpleEnvironment
import train as gradient_based_trainer # train.pyをインポート

def handle_agent(args):
    """自律エージェントの機能を処理する"""
    agent = AutonomousAgent()
    selected_model_info = agent.handle_task(
        task_description=args.task,
        unlabeled_data_path=args.unlabeled_data_path,
        force_retrain=args.force_retrain
    )
    if selected_model_info and args.prompt:
        print("\n" + "="*20 + " 🧠 INFERENCE " + "="*20)
        print(f"入力プロンプト: {args.prompt}")
        agent.run_inference(selected_model_info, args.prompt)
    elif not selected_model_info:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。")

def handle_planner(args):
    """階層的プランナーの機能を処理する"""
    planner = HierarchicalPlanner()
    final_result = planner.execute_task(
        task_request=args.request,
        context=args.context
    )
    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)

def handle_life_form(args):
    """デジタル生命体の機能を処理する"""
    life_form = DigitalLifeForm(project_root=".")
    life_form.awareness_loop(cycles=args.cycles)

def handle_evolution(args):
    """自己進化エージェントの機能を処理する"""
    agent = SelfEvolvingAgent(project_root=".", model_config_path=args.model_config)
    initial_metrics = {
        "accuracy": args.initial_accuracy,
        "avg_spikes_per_sample": args.initial_spikes
    }
    agent.run_evolution_cycle(
        task_description=args.task_description,
        initial_metrics=initial_metrics
    )

def handle_rl(args):
    """生物学的強化学習の機能を処理する"""
    # run_rl_agent.py のロジックをここに統合
    import torch
    from tqdm import tqdm

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    env = SimpleEnvironment(pattern_size=args.pattern_size, device=device)
    agent = ReinforcementLearnerAgent(input_size=args.pattern_size, output_size=args.pattern_size, device=device)
    
    progress_bar = tqdm(range(args.episodes))
    total_reward = 0.0

    for episode in progress_bar:
        state = env.reset()
        action = agent.get_action(state)
        _, reward, _ = env.step(action)
        agent.learn(reward)
        total_reward += reward
        avg_reward = total_reward / (episode + 1)
        progress_bar.set_postfix({"Avg Reward": f"{avg_reward:.3f}"})
    
    print(f"\n✅ 学習完了。最終的な平均報酬: {total_reward / args.episodes:.4f}")

def handle_train(args):
    """勾配ベース学習の機能を処理する"""
    # train.py の main() 関数を呼び出す
    # train.py内のDIコンテナ設定がそのまま利用される
    print("🔧 勾配ベースの学習プロセスを開始します...")
    
    # train.pyのmain関数が引数をパースするため、sys.argvを一時的に書き換える
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + args.train_args
    gradient_based_trainer.main()
    sys.argv = original_argv # 元に戻す

def main():
    parser = argparse.ArgumentParser(
        description="Project SNN: 統合CLIツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="実行する機能を選択")

    # --- Agent Subcommand ---
    parser_agent = subparsers.add_parser("agent", help="自律エージェントを操作して単一タスクを実行")
    parser_agent.add_argument("solve", help="指定されたタスクを解決します")
    parser_agent.add_argument("--task", type=str, required=True, help="タスクの自然言語説明 (例: '感情分析')")
    parser_agent.add_argument("--prompt", type=str, help="推論を実行する場合の入力プロンプト")
    parser_agent.add_argument("--unlabeled_data_path", type=str, help="新規学習時に使用するデータパス")
    parser_agent.add_argument("--force_retrain", action="store_true", help="モデル登録簿を無視して強制的に再学習")
    parser_agent.set_defaults(func=handle_agent)

    # --- Planner Subcommand ---
    parser_planner = subparsers.add_parser("planner", help="高次認知プランナーを操作して複雑なタスクを実行")
    parser_planner.add_argument("execute", help="複雑なタスク要求を実行します")
    parser_planner.add_argument("--request", type=str, required=True, help="タスク要求 (例: '記事を要約して感情を分析')")
    parser_planner.add_argument("--context", type=str, required=True, help="処理対象のデータ")
    parser_planner.set_defaults(func=handle_planner)

    # --- Life Form Subcommand ---
    parser_life = subparsers.add_parser("life-form", help="デジタル生命体の自律ループを開始")
    parser_life.add_argument("start", help="意識ループを開始します")
    parser_life.add_argument("--cycles", type=int, default=5, help="実行する意識サイクルの回数")
    parser_life.set_defaults(func=handle_life_form)

    # --- Evolution Subcommand ---
    parser_evo = subparsers.add_parser("evolve", help="自己進化サイクルを実行")
    parser_evo.add_argument("run", help="自己進化サイクルを1回実行します")
    parser_evo.add_argument("--task_description", type=str, required=True, help="自己評価の起点となるタスク説明")
    parser_evo.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="進化対象のモデル設定ファイル")
    parser_evo.add_argument("--initial_accuracy", type=float, default=0.75, help="自己評価のための初期精度")
    parser_evo.add_argument("--initial_spikes", type=float, default=1500.0, help="自己評価のための初期スパイク数")
    parser_evo.set_defaults(func=handle_evolution)
    
    # --- Reinforcement Learning Subcommand ---
    parser_rl = subparsers.add_parser("rl", help="生物学的強化学習を実行")
    parser_rl.add_argument("run", help="強化学習ループを開始します")
    parser_rl.add_argument("--episodes", type=int, default=100, help="学習エピソード数")
    parser_rl.add_argument("--pattern_size", type=int, default=10, help="環境のパターンサイズ")
    parser_rl.set_defaults(func=handle_rl)

    # --- Train Subcommand ---
    # 残りの引数をすべてtrain.pyに渡すためのパーサー
    parser_train = subparsers.add_parser("train", help="勾配ベースでSNNモデルを手動学習 (train.pyの引数を指定)")
    parser_train.add_argument('train_args', nargs=argparse.REMAINDER, help="train.pyに渡す引数 (例: --config ...)")
    parser_train.set_defaults(func=handle_train)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
