# ファイルパス: tests/cognitive_architecture/test_artificial_brain.py
# (新規作成)
#
# Title: 人工脳 統合テスト
#
# Description:
# - ArtificialBrainの主要な認知サイクルが、コンポーネントの連携を含め
#   エラーなく実行されることを確認する統合テスト。
# - DIコンテナ(BrainContainer)を使用して、依存関係が注入された
#   完全なArtificialBrainインスタンスを構築してテストする。

import sys
from pathlib import Path
import pytest

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.containers import BrainContainer

@pytest.fixture(scope="module")
def brain_container():
    """DIコンテナを初期化し、テストフィクスチャとして提供する。"""
    container = BrainContainer()
    # テスト用の設定をロードすることも可能
    container.config.from_yaml("configs/base_config.yaml")
    # RAGSystemの初回セットアップをシミュレート
    rag_system = container.rag_system()
    if not rag_system.vector_store:
        rag_system.setup_vector_store()
    return container

def test_artificial_brain_instantiation(brain_container: BrainContainer):
    """
    BrainContainerがArtificialBrainインスタンスを正常に構築できるかテストする。
    """
    brain = brain_container.artificial_brain()
    assert brain is not None
    assert brain.pfc is not None
    assert brain.hippocampus is not None
    assert brain.motor is not None
    print("✅ ArtificialBrainインスタンスの構築に成功しました。")

def test_cognitive_cycle_runs_without_errors(brain_container: BrainContainer):
    """
    run_cognitive_cycleがサンプル入力に対してエラーなく実行されるかテストする。
    """
    brain = brain_container.artificial_brain()
    
    test_input = "これはシステム全体の統合テストです。"
    
    try:
        brain.run_cognitive_cycle(test_input)
        print(f"✅ 認知サイクルが入力 '{test_input}' に対して正常に完了しました。")
    except Exception as e:
        pytest.fail(f"run_cognitive_cycleで予期せぬエラーが発生しました: {e}")