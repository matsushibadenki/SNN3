# /run_web_learning.py
# Title: Autonomous Web Learning Script
# Description: アイドル時にWebを巡回し、新しい知識を自律的に学習するサイクルを実行するスクリプト。

import argparse
from snn_research.tools.web_crawler import WebCrawler
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager

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
        default=10,
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
    distillation_manager = KnowledgeDistillationManager(
        base_config_path="configs/base_config.yaml",
        model_config_path="configs/models/small.yaml" # 新しい専門家はsmallモデルから開始
    )

    distillation_manager.run_on_demand_pipeline(
        task_description=args.topic,
        unlabeled_data_path=crawled_data_path,
        teacher_model_name="gpt2", # 教師モデルは設定可能
        force_retrain=True # 常に新しいデータで学習
    )

    print("\n🎉 自律的なWeb学習サイクルが完了しました。")
    print(f"  トピック「{args.topic}」に関する新しい専門家モデルが育成されました。")

if __name__ == "__main__":
    main()
