# /run_web_learning.py
# ファイルパス: matsushibadenki/snn3/SNN3-176e5ceb739db651438b22d74c0021f222858011/run_web_learning.py
# タイトル: Autonomous Web Learning Script
# 機能説明: DIコンテナからコンポーネントを取得する際のロジックを修正し、
#            Optimizerにモデルのパラメータが正しく渡されるようにすることで、TypeErrorを解消する。

import argparse
import os
import asyncio
from snn_research.tools.web_crawler import WebCrawler
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from app.containers import TrainingContainer # DIコンテナを利用

def main():
    """
    Webクローラーとオンデマンド学習パイプラインを連携させ、
    指定されたトピックに関する専門家モデルを自律的に生成する。
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Web Learning Framework",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="学習させたいトピック（タスク名として使用）。\n例: '最新のAI技術'"
    )
    parser.add_argument(
        "--start_url",
        type=str,
        required=True,
        help="情報収集を開始する起点となるURL。\n例: 'https://www.itmedia.co.jp/news/subtop/aiplus/'"
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=5, # デモ用に少なく設定
        help="収集するWebページの最大数。"
    )

    args = parser.parse_args()

    # --- ステップ1: Webクローリングによるデータ収集 ---
    print("\n" + "="*20 + " 🌐 Step 1: Web Crawling " + "="*20)
    crawler = WebCrawler()
    crawled_data_path = crawler.crawl(start_url=args.start_url, max_pages=args.max_pages)

    if not os.path.exists(crawled_data_path) or os.path.getsize(crawled_data_path) == 0:
        print("❌ データが収集できなかったため、学習を中止します。")
        return

    # --- ステップ2: オンデマンド知識蒸留による学習 ---
    print("\n" + "="*20 + " 🧠 Step 2: On-demand Learning " + "="*20)
    
    container = TrainingContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # 依存関係を正しい順序で構築する
    # 1. 生徒モデルをインスタンス化
    student_model = container.snn_model()

    # 2. モデルのパラメータを使ってオプティマイザをインスタンス化
    optimizer = container.optimizer(params=student_model.parameters())

    # 3. オプティマイザを使ってスケジューラをインスタンス化
    scheduler = container.scheduler(optimizer=optimizer)

    # 4. 構築した依存関係を渡してトレーナーをインスタンス化
    distillation_trainer = container.distillation_trainer(
        model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    distillation_manager = KnowledgeDistillationManager(
        student_model=student_model,
        trainer=distillation_trainer,
        teacher_model_name=container.config.training.gradient_based.distillation.teacher_model(),
        tokenizer_name=container.config.data.tokenizer_name(),
        model_registry=container.model_registry(),
        device=container.device()
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    asyncio.run(distillation_manager.run_on_demand_pipeline(
        task_description=args.topic,
        unlabeled_data_path=crawled_data_path,
        force_retrain=True
    ))

    print("\n🎉 自律的なWeb学習サイクルが完了しました。")
    print(f"  トピック「{args.topic}」に関する新しい専門家モデルが育成されました。")

if __name__ == "__main__":
    main()
