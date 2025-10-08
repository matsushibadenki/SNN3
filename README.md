# **Project SNN: A Predictive Digital Life Form (v4.2)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とし、**自律的デジタル生命体 (Autonomous Digital Life Form)** の創造を目指す研究開発フレームワークです。

最終目標は、静的なパターン認識の限界を超え、世界の動的なモデルを内的に構築し、**未来を予測し、その予測誤差を最小化する**という自己の存在理由に基づき、自律的に思考し、学習し、自己を改良するAIを実現することです。

このシステムは、単なるチャ-ットボットではなく、以下の高度な認知能力を備えた自律エージェントとして動作します。

* **デュアルコアSNNアーキテクチャ:** 脳の働きに着想を得た**予測符号化モデル**に加え、時間情報処理能力を最大化する**スパイキングトランスフォーマー**という、2つの最先端アーキテクチャをタスクに応じて切り替え可能です。  
* **オンデマンド学習:** 未知のタスクに直面した際、大規模言語モデルから知識を蒸留し、タスクに特化した超省エネルギーな「専門家SNN」を自律的に生成します。  
* **自己認識と計画立案:** 自身の能力（学習済み専門家モデル）を把握し、学習済みのプランナーSNNを用いて、複雑なタスクをサブタスクに分解し、最適な実行計画を動的に推論します。  
* **アーキテクチャレベルの自己進化:** 自らの性能を評価し、表現力不足と判断した場合、自身のソースコード（モデルの層数や次元数）を自律的に修正し、より強力なアーキテクチャへと進化します。  
* **行動を通じた学習（強化学習）:** バックプロパゲーションを使わない生物学的な学習則（報酬変調型STDP）を用い、環境との試行錯誤から直接スキルを学習する能力を持ちます。

## **2\. システムアーキテクチャ**

本システムの認知アーキテクチャは、複数の専門コンポーネントが階層的に連携することで実現されています。

graph TD  
    CLI\[snn-cli.py 統合CLIツール\]

    subgraph Phase6\[Phase 6: 自律的存在\]  
        LifeForm\[life-form start\]  
        Motivation\[Physics-Aware IntrinsicMotivationSystem\]  
        Decision{探求 or 自己改善?}

        LifeForm \--\> Motivation  
        Motivation \--\> Decision

        subgraph Exploration\[探求\]  
            ExploreDecision{思考 or 行動?}  
            Planner\[planner execute 思考による探求\]  
            RLAgent\[rl run 行動による探求\]

            Decision \--\>|探求 退屈| ExploreDecision  
            ExploreDecision \--\>|思考| Planner  
            ExploreDecision \--\>|行動| RLAgent  
        end

        subgraph SelfImprove\[自己改善\]  
            Evolution\[evolve run 自己アーキテクチャを分析修正\]  
            Benchmark\[run\_benchmark.py\]

            Decision \--\>|改善 現状満足| Evolution  
            Evolution \--\> Benchmark  
        end  
    end

    subgraph ExecCore\[Phase 3-5: 高次認知実行コア\]  
        Planner \--\> Workspace\[GlobalWorkspace\]  
        Workspace \--\>|サブタスク実行| Specialist\[専門家SNN\]  
        Workspace \--\>|知識が必要| RAG\[RAGSystem ベクトルストア検索\]  
    end

    subgraph LearnInfer\[Phase 0-2.5: 学習推論エンジン\]  
        subgraph Training\[学習\]  
            TrainPy\[train.py\]  
            SNN\_Core\[SpikingTransformer or BreakthroughSNN\]  
            MetaSNN\[MetaCognitiveSNN SNAKE\]  
            TrainPy \--\> SNN\_Core  
            SNN\_Core \--\> MetaSNN  
        end  
        subgraph Inference\[推論\]  
            Specialist \--\> Deployment\[SNNInferenceEngine\]  
        end  
    end

    subgraph OnDemand\[オンデマンド学習\]  
        Agent\[agent solve\]  
        Agent \--\>|未知のタスク| TrainPy  
        Agent \--\>|モデル検索| Registry\[ModelRegistry\]  
    end

    CLI \--\> LifeForm  
    CLI \--\> Planner  
    CLI \--\> RLAgent  
    CLI \--\> Evolution  
    CLI \--\> Agent  
    CLI \--\> TrainPy

    style CLI fill:\#b39ddb,stroke:\#333,stroke-width:3px  
    style LifeForm fill:\#cde4ff,stroke:\#333,stroke-width:1px  
    style Planner fill:\#ffe4c4,stroke:\#333,stroke-width:1px  
    style RLAgent fill:\#ffe4c4,stroke:\#333,stroke-width:1px  
    style Agent fill:\#d4edda,stroke:\#333,stroke-width:1px  
    style TrainPy fill:\#f8d7da,stroke:\#333,stroke-width:1px  
    style SNN\_Core fill:\#fff2cd,stroke:\#333,stroke-width:2px

## **3\. システムの実行方法**

### **ステップ1: 環境設定**

まず、必要なPythonライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: 統合CLIツール snn-cli.py の使い方**

本プロジェクトの全ての機能は、snn-cli.py という単一のコマンドから実行できます。-h または \--help を付けて実行すると、利用可能な機能の一覧が表示されます。

python snn-cli.py \--help

### **A) 迅速な機能テスト（数分で完了）**

#### **A-1. オンデマンド学習の動作確認**

**目的:** agent solveコマンドが、**小規模なサンプルデータ**を使ってエラーなく学習・推論サイクルを完了できるかを確認します。

python snn-cli.py agent solve \\  
    \--task "高速テスト" \\  
    \--prompt "これはテストです。" \\  
    \--unlabeled-data data/sample\_data.jsonl \\  
    \--force-retrain

**Note:** このテストはあくまでシステムの動作確認用です。小規模データのため、意味のある応答は生成されません。

#### **A-2. 手動でのモデル学習**

**目的:** gradient-trainコマンドが、指定された設定でエラーなく学習を完了できるかを確認します。

python snn-cli.py gradient-train \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=3"

### **B) SNN能力向上のための本格的な学習（時間がかかります）**

#### **B-1. Webからの自律学習**

**目的:** AIに新しいトピックをWebから自律的に学習させ、その知識に基づいた専門家モデルを生成させます。

python run\_web\_learning.py \\  
    \--topic "最新の半導体技術" \\  
    \--start\_url "\[https://pc.watch.impress.co.jp/\](https://pc.watch.impress.co.jp/)" \\  
    \--max\_pages 10

#### **B-2. 大規模データセットによるオンデマンド学習**

**目的:** wikitext-103（100万行以上のテキスト）を使い、汎用的な言語能力を持つ専門家モデルを育成します。**AIの応答品質を本格的に向上させるには、このコマンドの実行が必要です。**

**ステップ 1: 大規模データセットの準備（初回のみ）**

python scripts/data\_preparation.py

**ステップ 2: 本格的な学習の実行**

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--force-retrain

**Note:** この学習はマシンスペックにより数時間以上かかる可能性があります。

#### **B-3. 学習済みモデルとの対話**

**目的:** 上記の本格的な学習で育成した専門家モデルを呼び出して対話します。

\# 「汎用言語モデル」との対話  
python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--prompt "SNNとは何ですか？"

\# 「最新の半導体技術」専門家との対話  
python snn-cli.py agent solve \\  
    \--task "最新の半導体技術" \\  
    \--prompt "3nmプロセスとは何ですか？"

### **C) その他の高度な機能**

その他の高度な機能については、SNN開発：プロジェクト機能テスト コマンド一覧.md をご参照ください。

## **4\. プロジェクト構造**

snn3/  
├── app/                  \# UIアプリケーションとDIコンテナ  
├── configs/              \# 設定ファイル (base, models/\*.yaml)  
├── data/                 \# 学習用データセット  
├── doc/                  \# ドキュメント  
├── precomputed\_data/     \# (自動生成) 知識蒸留用の中間データ  
├── runs/                 \# (自動生成) 学習ログ、チェックポイント、モデル登録簿  
├── scripts/              \# データ準備やベンチマークなどの補助スクリプト  
├── snn\_research/         \# SNNコア研究開発コード  
│   ├── agent/            \# 各種エージェント (自律、自己進化、生命体、強化学習)  
│   ├── benchmark/        \# SNN vs ANN 性能評価  
│   ├── cognitive\_architecture/ \# 高次認知機能 (プランナー、物理評価器等)  
│   ├── core/             \# SNNモデル (BreakthroughSNN, SpikingTransformer)  
│   ├── data/             \# データセット定義  
│   ├── deployment.py     \# 推論エンジン  
│   ├── distillation/     \# 知識蒸留とモデル登録簿  
│   ├── learning\_rules/   \# 生物学的学習則 (STDPなど)  
│   ├── rl\_env/           \# 強化学習環境  
│   ├── tools/            \# 外部ツール (Webクローラーなど)  
│   └── training/         \# Trainerと損失関数  
├── snn-cli.py            \# ✨ 統合CLIツール  
├── train.py              \# 勾配ベース学習の実行スクリプト (CLIから呼び出される)  
├── run\_web\_learning.py   \# ✨ Webからの自律学習実行スクリプト  
└── requirements.txt      \# 必要なライブラリ  
