# **プロジェクト機能テスト コマンド一覧**

このドキュメントは、プロジェクトの各機能をテストするための簡単なコマンドをまとめたものです。

## **1\. Webからの自律学習と推論 (run\_web\_learning.py)**

AIに新しいトピックをWebから学習させ、その知識について質問します。

### **学習**

python run\_web\_learning.py \\  
    \--topic "最新のAI技術" \\  
    \--start\_url "\[https://www.itmedia.co.jp/news/subtop/aiplus/\](https://www.itmedia.co.jp/news/subtop/aiplus/)" \\  
    \--max\_pages 5

### **推論**

python snn-cli.py agent solve \\  
    \--task "最新のAI技術" \\  
    \--prompt "SNNとは何ですか？" \\  
    \--min\_accuracy 0.4

## **2\. ローカルデータからのオンデマンド学習と推論 (agent solve)**

手元のデータを使って、新しい「専門家モデル」を育成し、タスクを実行させます。

### **学習**

python snn-cli.py agent solve \\  
    \--task "文章要約" \\  
    \--unlabeled\_data\_path data/sample\_data.jsonl

### **推論**

python snn-cli.py agent solve \\  
    \--task "文章要約" \\  
    \--prompt "SNNは、生物の神経系における情報の伝達と処理のメカニズムを模倣したニューラルネットワークの一種である。"

## **3\. 手動でのモデル学習 (gradient-train)**

特定のモデル設定（例：large.yaml）を使って、手動でモデルを学習させます。

### **学習**

python snn-cli.py gradient-train \\  
    \--model\_config configs/models/large.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=5"

*学習後、モデルは runs/snn\_experiment/best\_model.pth に保存されます。*

### **推論 (対話UI)**

python app/main.py \\  
    \--model\_config configs/models/large.yaml \\  
    \--model\_path runs/snn\_experiment/best\_model.pth

## **4\. 複雑なタスクの計画と実行 (planner execute)**

複数のステップが必要な複雑なタスクをプランナーに依頼します。

### **実行**

python snn-cli.py planner execute \\  
    \--request "この記事を要約して、その内容の感情を分析してください。" \\  
    \--context "SNNは非常にエネルギー効率が高いことで知られているが、その性能はまだANNに及ばない点もある。"

## **5\. 自己進化 (evolve run)**

AIに自身の性能を評価させ、モデルのアーキテクチャを改善させます。

### **実行**

python snn-cli.py evolve run \\  
    \--task\_description "高難度タスク" \\  
    \--initial\_accuracy 0.4 \\  
    \--model\_config "configs/models/small.yaml"

*実行後、configs/models/small\_evolved\_v2.yaml のような新しい設定ファイルが生成されます。*

## **6\. 強化学習 (rl run)**

生物学的な学習ルール（報酬変調型STDP）を用いて、AIが試行錯誤から学習するプロセスを実行します。

### **実行**

python snn-cli.py rl run \--episodes 100

## **7\. デジタル生命体の自律ループ (life-form start)**

AIが自らの「好奇心」や「退屈」といった内部状態に基づき、自律的に活動するループを開始します。

### **実行**

python snn-cli.py life-form start \--cycles 10

## **8\. 対話UIの起動 (app/main.py)**

学習済みのモデルとチャット形式で対話できるWeb UIを起動します。

### **実行**

python app/main.py \--model\_config configs/models/medium.yaml

## **9\. ベンチマークの実行 (run\_benchmark.py)**

SNNとANNの性能を比較評価します。

### **実行**

python scripts/run\_benchmark.py \--task sst2  
