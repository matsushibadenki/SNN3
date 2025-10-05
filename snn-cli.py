# /snn-cli.py
# Title: 統合CLIツール
# Description: プロジェクトの全機能をサブコマンド形式で実行するための統一インターフェース。
# mypyエラー修正: 存在しないSpikingDatasetの代わりにダミークラスを追加。
#                 型エラー(DictConfig, int/float)を修正。
#                 重複するtrain関数を削除。
# mypyエラー修正: 各エージェント・プランナーの呼び出しと引数を修正。
# mypyエラー修正: ModelRegistryのインスタンス化を具象クラスに変更。
# 改善点: RAGSystemをHierarchicalPlannerに注入するように修正。
# mypyエラー修正: asyncio.run() を使って非同期関数を呼び出すように修正。

import argparse
import sys
import asyncio
from pathlib import Path
import os
from typing import Tuple, cast, Dict, Any
import torch
from torch.utils.data import Dataset
import typer
from omegaconf import OmegaConf, DictConfig

from snn_research.core.snn_core import SNNCore
from snn_research.training.trainers import BPTTTrainer


# --- プロジェクトルートをPythonパスに追加 ---
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


app = typer.Typer()

@app.command(name="train-basic", help="[簡易版] BPTTベースの基本的なSNNモデル学習を実行します。より高度な機能（分散学習、詳細な設定）が必要な場合は `gradient-train` コマンドを使用してください。")
def train_basic_command(
    config_path: str = typer.Option("configs/base_config.yaml", help="Path to the base config file."),
    model_config_path: str = typer.Option(..., help="Path to the model config file (e.g., configs/models/spiking_transformer.yaml)."),
    output_dir: str = typer.Option("models/", help="Directory to save the trained model.")
):
    """SNNモデルの学習を実行する。"""
    print("--- Starting Basic Training ---")

    base_cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.load(model_config_path)
    cfg = cast(DictConfig, OmegaConf.merge(base_cfg, model_cfg))
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))

    # 簡易データセットのためのダミークラスをローカルで定義
    class _SpikingDataset(Dataset):
        def __init__(self, num_samples: int, sequence_length: int, vocab_size: int):
            self.num_samples = num_samples
            self.sequence_length = sequence_length
            self.vocab_size = vocab_size

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            data = torch.randint(0, self.vocab_size, (self.sequence_length,))
            targets = torch.randint(0, self.vocab_size, (self.sequence_length,))
            return data, targets
            
    vocab_size = cfg.model.get("vocab_size", cfg.data.get("max_vocab_size", 50000))
    dataset = _SpikingDataset(num_samples=100, sequence_length=cfg.model.get("time_steps", 32), vocab_size=vocab_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.training.batch_size)
    print(f"Dataset prepared with {len(dataset)} samples.")

    model = SNNCore(cfg, vocab_size=vocab_size)
    print("Model initialized:")
    print(model)

    trainer = BPTTTrainer(model, cfg)
    print("Trainer initialized.")

    print("--- Training Loop ---")
    epochs = cfg.training.get("epochs", 1) 
    for epoch in range(epochs):
        total_loss: float = 0.0
        for i, (data, targets) in enumerate(dataloader):
            loss = trainer.train_step(data, targets)
            total_loss += loss
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss:.4f}")
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    model_name = cfg.model.get("architecture_type", cfg.model.get("type", "snn_model"))
    save_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Finished ---")
    print(f"Model saved to {save_path}")
    
    
def handle_agent(args: argparse.Namespace) -> None:
    """自律エージェントの機能を処理する"""
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
        accuracy_threshold=args.min_accuracy,
        energy_budget=args.max_spikes
    )
    
    # 非同期関数を asyncio.run で実行
    selected_model_info = asyncio.run(agent.handle_task(
        task_description=args.task,
        unlabeled_data_path=args.unlabeled_data_path,
        force_retrain=args.force_retrain
    ))
    
    if selected_model_info and args.prompt:
        print("\n" + "="*20 + " 🧠 INFERENCE " + "="*20)
        print(f"入力プロンプト: {args.prompt}")
        asyncio.run(agent.run_inference(selected_model_info, args.prompt))
    elif not selected_model_info:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。")


def handle_planner(args: argparse.Namespace) -> None:
    """階層的プランナーの機能を処理する"""
    model_registry = SimpleModelRegistry()
    rag_system = RAGSystem()
    planner = HierarchicalPlanner(model_registry=model_registry, rag_system=rag_system)
    
    final_result = planner.execute_task(
        task_request=args.request,
        context=args.context
    )
    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)

def handle_life_form(args: argparse.Namespace) -> None:
    """デジタル生命体の機能を処理する"""
    life_form = DigitalLifeForm()
    life_form.awareness_loop(cycles=args.cycles)

def handle_evolution(args: argparse.Namespace) -> None:
    """自己進化エージェントの機能を処理する"""
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
        model_config_path=args.model_config
    )
    initial_metrics = {
        "accuracy": args.initial_accuracy,
        "avg_spikes_per_sample": args.initial_spikes
    }
    agent.run_evolution_cycle(
        task_description=args.task_description,
        initial_metrics=initial_metrics
    )

def handle_rl(args: argparse.Namespace) -> None:
    """生物学的強化学習の機能を処理する"""
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

def handle_gradient_train(args: argparse.Namespace) -> None:
    """勾配ベース学習の機能を処理する (train.pyを呼び出す)"""
    print("🔧 勾配ベースの学習プロセスを開始します...")
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + args.train_args
    gradient_based_trainer.main()
    sys.argv = original_argv

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project SNN: 統合CLIツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="実行する機能を選択")

    parser_agent = subparsers.add_parser("agent", help="自律エージェントを操作して単一タスクを実行")
    parser_agent.add_argument("solve", help="指定されたタスクを解決します")
    parser_agent.add_argument("--task", type=str, required=True, help="タスクの自然言語説明 (例: '感情分析')")
    parser_agent.add_argument("--prompt", type=str, help="推論を実行する場合の入力プロンプト")
    parser_agent.add_argument("--unlabeled_data_path", type=str, help="新規学習時に使用するデータパス")
    parser_agent.add_argument("--force_retrain", action="store_true", help="モデル登録簿を無視して強制的に再学習")
    parser_agent.add_argument("--min_accuracy", type=float, default=0.6, help="専門家モデルを選択するための最低精度要件 (デフォルト: 0.6)")
    parser_agent.add_argument("--max_spikes", type=float, default=10000.0, help="専門家モデルを選択するための平均スパイク数上限 (デフォルト: 10000.0)")
    parser_agent.set_defaults(func=handle_agent)

    parser_planner = subparsers.add_parser("planner", help="高次認知プランナーを操作して複雑なタスクを実行")
    parser_planner.add_argument("execute", help="複雑なタスク要求を実行します")
    parser_planner.add_argument("--request", type=str, required=True, help="タスク要求 (例: '記事を要約して感情を分析')")
    parser_planner.add_argument("--context", type=str, required=True, help="処理対象のデータ")
    parser_planner.set_defaults(func=handle_planner)

    parser_life = subparsers.add_parser("life-form", help="デジタル生命体の自律ループを開始")
    parser_life.add_argument("start", help="意識ループを開始します")
    parser_life.add_argument("--cycles", type=int, default=5, help="実行する意識サイクルの回数")
    parser_life.set_defaults(func=handle_life_form)

    parser_evo = subparsers.add_parser("evolve", help="自己進化サイクルを実行")
    parser_evo.add_argument("run", help="自己進化サイクルを1回実行します")
    parser_evo.add_argument("--task_description", type=str, required=True, help="自己評価の起点となるタスク説明")
    parser_evo.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="進化対象のモデル設定ファイル")
    parser_evo.add_argument("--initial_accuracy", type=float, default=0.75, help="自己評価のための初期精度")
    parser_evo.add_argument("--initial_spikes", type=float, default=1500.0, help="自己評価のための初期スパイク数")
    parser_evo.set_defaults(func=handle_evolution)
    
    parser_rl = subparsers.add_parser("rl", help="生物学的強化学習を実行")
    parser_rl.add_argument("run", help="強化学習ループを開始します")
    parser_rl.add_argument("--episodes", type=int, default=100, help="学習エピソード数")
    parser_rl.add_argument("--pattern_size", type=int, default=10, help="環境のパターンサイズ")
    parser_rl.set_defaults(func=handle_rl)

    parser_train = subparsers.add_parser("gradient-train", help="勾配ベースでSNNモデルを手動学習 (train.pyの引数を指定)")
    parser_train.add_argument('train_args', nargs=argparse.REMAINDER, help="train.pyに渡す引数 (例: --config ...)")
    parser_train.set_defaults(func=handle_gradient_train)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
         app()
    else:
         main()
