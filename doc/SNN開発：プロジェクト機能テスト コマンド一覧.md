# **プロジェクト機能テスト コマンド一覧 (v2.0)**

このドキュメントは、プロジェクトの各機能をテストするための主要なコマンドをまとめたものです。コマンドはすべて統合CLIツール snn-cli.py または専用の実行スクリプトを通じて実行します。

## **A) 迅速な機能テスト（数分で完了）**

**目的:** システム全体がエラーなく動作することを確認するための、小規模データを用いた基本的なテスト。

### **1\. オンデマンド学習の動作確認**

agent solveコマンドが、サンプルデータを使って学習と推論のサイクルを完了できるかを確認します。

python snn-cli.py agent solve \\  
    \--task "高速テスト" \\  
    \--prompt "これはテストです。" \\  
    \--unlabeled-data data/sample\_data.jsonl \\  
    \--force-retrain

**Note:** このテストはシステムの動作確認用です。小規模データのため、AIは意味のある応答を生成できません。

### **2\. 手動での勾配ベース学習**

gradient-trainコマンドが、指定された設定でエラーなく短時間の学習を完了できるかを確認します。

python snn-cli.py gradient-train \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=3"

## **B) 主要機能テスト（学習と評価）**

**目的:** SNNモデルの主要な学習パラダイムを実行し、その成果を確認します。

### **3\. 生物学的強化学習 (ロードマップ フェーズ2完了検証)**

run\_rl\_agent.pyを使用し、エージェントが複数ステップのGridWorld環境でタスクを学習できることを検証します。学習後、**学習曲線グラフ (rl\_learning\_curve.png)** と**訓練済みモデル (trained\_rl\_agent.pth)** がruns/rl\_results/ディレクトリに保存されます。

python run\_rl\_agent.py \--episodes 1000 \--grid\_size 5 \--max\_steps 50

### **4\. 大規模データセットによるオンデマンド学習**

wikitext-103を使い、汎用的な言語能力を持つ専門家モデルを育成します。AIの応答品質を本格的に向上させるには、このコマンドの実行が必要です。

**ステップ 1: 大規模データセットの準備（初回のみ）**

python scripts/data\_preparation.py

**ステップ 2: 本格的な学習の実行**

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--force-retrain

### **5\. 学習済みモデルとの対話**

上記の学習で育成した専門家モデルを呼び出して対話します。

\# 「汎用言語モデル」との対話  
python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--prompt "SNNとは何ですか？"

## **C) 高度な認知・自律機能テスト**

**目的:** 自己進化やマルチエージェント協調など、より高度なシステムの動作を確認します。

### **6\. Webからの自律学習**

AIに新しいトピックをWebから自律的に学習させ、その知識に基づいた専門家モデルを生成させます。

python run\_web\_learning.py \\  
    \--topic "最新の半導体技術" \\  
    \--start\_url "\[https://pc.watch.impress.co.jp/\](https://pc.watch.impress.co.jp/)" \\  
    \--max\_pages 10

### **7\. 自己進化**

エージェントが自身の性能を評価し、アーキテクチャや学習パラメータを改善するプロセスをテストします。

python snn-cli.py evolve run \\  
    \--task\_description "高難度タスク" \\  
    \--initial\_accuracy 0.4 \\  
    \--model\_config "configs/models/small.yaml" \\  
    \--training\_config "configs/base\_config.yaml"

### **8\. デジタル生命体の自律ループ**

AIが外部からの指示なしに、内発的動機に基づいて自律的に思考・学習するループを開始します。

python snn-cli.py life-form start \--cycles 10

### **9\. マルチエージェントによる協調的タスク解決**

複数のエージェントが協調して単一の目標を解決する創発的システムを起動します。

python snn-cli.py emergent-system execute \\  
    \--goal "最新のAIトレンドを調査し、その内容を要約する"

### **10\. 対話UIの起動**

snn-cli.pyからGradioベースの対話UIを起動します。

\# 標準UIの起動  
python snn-cli.py ui start \--model\_config configs/models/medium.yaml

\# LangChain連携版UIの起動  
python snn-cli.py ui start-langchain \--model\_config configs/models/medium.yaml  
