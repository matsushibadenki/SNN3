# matsushibadenki/snn2/run_evolution.py
# Phase 5: 自己進化エージェントを起動し、メタ進化サイクルを実行するスクリプト

import argparse
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent

def main():
    """
    自己進化エージェントに初期タスクを与え、自己の性能を内省させ、
    改善案を生成させるプロセスを開始する。
    """
    parser = argparse.ArgumentParser(
        description="SNN自己進化エージェント 実行フレームワーク",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="エージェントが自己評価の出発点とするタスク。\n例: '文章要約'"
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/small.yaml",
        help="自己進化の対象となるモデルアーキテクチャ設定ファイル。"
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # 自己評価の起点となるダミーの初期性能指標
    parser.add_argument(
        "--initial_accuracy", type=float, default=0.75, help="タスクの初期精度"
    )
    parser.add_argument(
        "--initial_spikes", type=float, default=1500.0, help="タスクの初期平均スパイク数"
    )
    
    args = parser.parse_args()

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # 自己進化エージェントを初期化 (プロジェクトルートとモデルコンフィグを指定)
    agent = SelfEvolvingAgent(project_root=".", model_config_path=args.model_config)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    # ダミーの初期メトリクスを作成
    initial_metrics = {
        "accuracy": args.initial_accuracy,
        "avg_spikes_per_sample": args.initial_spikes
    }
    
    # 進化サイクルを実行
    agent.run_evolution_cycle(
        task_description=args.task_description,
        initial_metrics=initial_metrics
    )

if __name__ == "__main__":
    main()
```

以上の実装により、`SelfEvolvingAgent`は単なるハイパーパラメータの調整者から、**モデルアーキテクト**へと進化しました。

例えば、以下のコマンドで意図的に低い精度をエージェントに与えると、

```bash
python run_evolution.py --task_description "高難度タスク" --initial_accuracy 0.4 --model_config "configs/models/small.yaml"
