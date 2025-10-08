# **プロジェクト機能テスト コマンド一覧 (更新版)**

このドキュメントは、プロジェクトの各機能をテストするためのコマンドをまとめたものです。

## **A) 迅速な機能テスト（数分で完了）**

### **1\. オンデマンド学習の動作確認**

**目的:** agent solveコマンドが、**小規模なサンプルデータ**を使ってエラーなく学習・推論サイクルを完了できるかを確認します。

python snn-cli.py agent solve \\  
    \--task "高速テスト" \\  
    \--prompt "これはテストです。" \\  
    \--unlabeled-data data/sample\_data.jsonl \\  
    \--force-retrain

**Note:** このテストはあくまでシステムの動作確認用です。小規模データのため、意味のある応答は生成されません。

### **2\. 手動でのモデル学習**

**目的:** gradient-trainコマンドが、指定された設定でエラーなく学習を完了できるかを確認します。

python snn-cli.py gradient-train \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=3"

## **B) SNN能力向上のための本格的な学習（時間がかかります）**

### **3\. Webからの自律学習**

**目的:** AIに新しいトピックをWebから自律的に学習させ、その知識に基づいた専門家モデルを生成させます。

python run\_web\_learning.py \\  
    \--topic "最新の半導体技術" \\  
    \--start\_url "\[https://pc.watch.impress.co.jp/\](https://pc.watch.impress.co.jp/)" \\  
    \--max\_pages 10

### **4\. 大規模データセットによるオンデマンド学習**

**目的:** wikitext-103（100万行以上のテキスト）を使い、汎用的な言語能力を持つ専門家モデルを育成します。**AIの応答品質を本格的に向上させるには、このコマンドの実行が必要です。**

**ステップ 1: 大規模データセットの準備（初回のみ）**

python scripts/data\_preparation.py

**ステップ 2: 本格的な学習の実行**

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--force-retrain

**Note:** この学習はマシンスペックにより数時間以上かかる可能性があります。学習後、以下のコマンドで対話ができます。

### **5\. 学習済みモデルとの対話**

**目的:** 上記の本格的な学習で育成した専門家モデルを呼び出して対話します。

\# 「汎用言語モデル」との対話  
python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--prompt "SNNとは何ですか？"

\# 「最新の半導体技術」専門家との対話  
python snn-cli.py agent solve \\  
    \--task "最新の半導体技術" \\  
    \--prompt "3nmプロセスとは何ですか？"

## **C) その他の高度な機能**

### **6\. 複雑なタスクの計画と実行**

python snn-cli.py planner execute \\  
    \--request "この記事を要約して、その内容の感情を分析してください。" \\  
    \--context "SNNは非常にエネルギー効率が高いことで知られているが、その性能はまだANNに及ばない点もある。"

### **7\. 自己進化**

python snn-cli.py evolve run \\  
    \--task\_description "高難度タスク" \\  
    \--initial\_accuracy 0.4 \\  
    \--model\_config "configs/models/small.yaml"

### **8\. 強化学習**

python snn-cli.py rl run \--episodes 100

### **9\. デジタル生命体の自律ループ**

python snn-cli.py life-form start \--cycles 10

### **10\. 対話UIの起動**

python app/main.py \--model\_config configs/models/medium.yaml  
