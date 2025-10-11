# ファイルパス: run_brain_simulation.py
# (更新)
# 修正: DIコンテナ(BrainContainer)を使用して依存関係を構築し、
#       ArtificialBrainインスタンスを取得するようにリファクタリング。

import sys
from pathlib import Path
import time

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

from app.containers import BrainContainer

def main():
    """
    DIコンテナを使って人工脳を初期化し、シミュレーションを実行する。
    """
    # 1. DIコンテナを初期化
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml") # 必要に応じて設定をロード

    # 2. コンテナから完成品の人工脳インスタンスを取得
    brain = container.artificial_brain()

    # 3. シミュレーションの実行
    inputs = [
        "素晴らしい発見だ！これは成功に繋がるだろう。",
        "エラーが発生しました。システムに問題があるようです。",
        "今日は穏やかな一日だ。"
    ]

    for text_input in inputs:
        brain.run_cognitive_cycle(text_input)
        time.sleep(1) # 各サイクルの間に少し待機

if __name__ == "__main__":
    main()
