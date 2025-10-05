# matsushibadenki/snn3/run_planner.py
# Phase 3: 高次認知アーキテクチャの実行インターフェース
# 改善点: プランナーに必要な依存関係(RAGSystem, ModelRegistry)を初期化するように修正。

import argparse
import asyncio
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.distillation.model_registry import SimpleModelRegistry
from app.containers import AgentContainer

def main():
    """
    階層的思考プランナーに複雑なタスクを依頼し、
    複数の専門家SNNを連携させた問題解決を実行させる。
    """
    parser = argparse.ArgumentParser(
        description="SNN高次認知アーキテクチャ 実行フレームワーク",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--task_request",
        type=str,
        required=True,
        help="エージェントに解決させたい、自然言語による複雑なタスク要求。\n例: 'この記事を要約して、その内容の感情を分析してください。'"
    )
    parser.add_argument(
        "--context_data",
        type=str,
        required=True,
        help="タスク処理の対象となる文脈データ（文章や質問など）。"
    )

    args = parser.parse_args()

    # DIコンテナを初期化し、依存関係が注入済みのプランナーを取得
    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")
    # model_configは直接プランナーに関係ないが、念のため読み込む
    container.config.from_yaml("configs/models/small.yaml") 
    
    planner = container.hierarchical_planner()
    
    # RAGの知識ベースを構築（存在しない場合）
    rag_system = container.rag_system()
    if not rag_system.vector_store:
        print("知識ベースが存在しないため、初回構築を行います...")
        rag_system.setup_vector_store()


    # --- 依存関係の構築 ---
    model_registry = SimpleModelRegistry()
    rag_system = RAGSystem()
    # 知識ベースがなければ構築する
    if not rag_system.vector_store:
        print("知識ベースが存在しないため、初回構築を行います...")
        rag_system.setup_vector_store()

    # --- 階層的プランナーを初期化 ---
    planner = HierarchicalPlanner(
        model_registry=model_registry,
        rag_system=rag_system
    )

    # --- プランナーにタスク処理を依頼 ---
    final_result = planner.execute_task(
        task_request=args.task_request,
        context=args.context_data
    )

    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
        print("="*56)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。必要な専門家モデルが不足している可能性があります。")

if __name__ == "__main__":
    main()
