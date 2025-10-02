# **Project SNN: A Predictive Digital Life Form (v4.0)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とし、**自律的デジタル生命体 (Autonomous Digital Life Form)** の創造を目指す研究開発フレームワークです。

最終目標は、静的なパターン認識の限界を超え、世界の動的なモデルを内的に構築し、**未来を予測し、その予測誤差を最小化する**という自己の存在理由（自由エネルギー原理）に基づき、自律的に思考し、学習し、自己を改良するAIを実現することです。

このシステムは、単なるチャットボットではなく、以下の高度な認知能力を備えた自律エージェントとして動作します。

* **デュアルコアSNNアーキテクチャ:** 脳の働きに着想を得た**予測符号化モデル**に加え、時間情報処理能力を最大化する**スパイキングトランスフォーマー**という、2つの最先端アーキテクチャをタスクに応じて切り替え可能です。  
* **オンデマンド学習:** 未知のタスクに直面した際、大規模言語モデルから知識を蒸留し、タスクに特化した超省エネルギーな「専門家SNN」を自律的に生成します。  
* **自己認識と計画立案:** 自身の能力（学習済み専門家モデル）を把握し、**学習済みのプランナーSNN**を用いて、複雑なタスクをサブタスクに分解し、最適な実行計画を動的に推論します。  
* **アーキテクチャレベルの自己進化:** 自らの性能を評価し、表現力不足と判断した場合、自身のソースコード（**モデルの層数や次元数**）を自律的に修正し、より強力なアーキテクチャへと進化します。  
* **物理法則を考慮した好奇心:** 予測が上達することに加え、自身の内部状態が\*\*物理法則（エネルギー効率や滑らかさ）\*\*に合致すること自体を「報酬」と感じる、より高度な内発的動機付けを持ちます。  
* **行動を通じた学習（強化学習）:** 従来の思考（推論）による問題解決だけでなく、バックプロパゲーションを一切使わない生物学的な学習則（報酬変調型STDP）を用い、環境との**試行錯誤**から直接スキルを学習する能力を持ちます。

## **2\. システムアーキテクチャ**

本システムの認知アーキテクチャは、複数の専門コンポーネントが階層的に連携することで実現されています。

graph TD  
    %% Phase 6: 自律的存在 (Digital Life Form)  
    subgraph Phase6 \["Phase 6: 自律的存在 (Digital Life Form)"\]  
        direction LR  
        LifeForm\[/"run\_life\_form.py\<br/\>(DigitalLifeForm)"/\]  
        Motivation\[/"snn\_research/cognitive\_architecture\<br/\>(Physics-Aware IntrinsicMotivationSystem)"/\]  
        Decision{"探求 or 自己改善?"}

        LifeForm \--\> Motivation  
        Motivation \--\> Decision

        subgraph Exploration \["探求 (Exploration)"\]  
            direction TB  
            ExploreDecision{"思考 or 行動?"}  
            Planner\[/"run\_planner.py\<br/\>(HierarchicalPlanner)\<br/\>思考による探求"/\]  
            RLAgent\[/"run\_rl\_agent.py\<br/\>(ReinforcementLearnerAgent)\<br/\>行動による探求"/\]

            Decision \-- "探求 (退屈)" \--\> ExploreDecision  
            ExploreDecision \-- "思考" \--\> Planner  
            ExploreDecision \-- "行動" \--\> RLAgent  
        end

        subgraph SelfImprove \["自己改善 (Self-Improvement)"\]  
            direction TB  
            Evolution\[/"run\_evolution.py\<br/\>(SelfEvolvingAgent)\<br/\>自己アーキテクチャを分析・修正"/\]  
            Benchmark\[/"scripts/run\_benchmark.py"/\]

            Decision \-- "改善 (現状満足)" \--\> Evolution  
            Evolution \--\> Benchmark  
        end  
    end

    %% Phase 3-5: 高次認知実行コア  
    subgraph ExecCore \["Phase 3-5: 高次認知実行コア (Cognitive Execution Core)"\]  
        direction TB  
        Planner \--\> Workspace\[/"snn\_research/cognitive\_architecture\<br/\>(GlobalWorkspace)"/\]  
        Workspace \--\>|サブタスク実行| Specialist\[/"専門家SNN\<br/\>(Specialist SNN)"/\]  
        Workspace \--\>|知識が必要| RAG\[/"snn\_research/cognitive\_architecture\<br/\>(RAGSystem)\<br/\>ベクトルストア検索"/\]  
    end

    %% Phase 0-2.5: 学習・推論エンジン  
    subgraph LearnInfer \["Phase 0-2.5: 学習・推論エンジン (Learning & Inference Engine)"\]  
        direction TB  
        subgraph Training \["学習 (Training)"\]  
            TrainPy\[/"train.py"/\]  
            SNN\_Core\["snn\_research/core\<br/\>(SpikingTransformer or BreakthroughSNN)"\]  
            MetaSNN\[/"snn\_research/cognitive\_architecture\<br/\>(MetaCognitiveSNN \- SNAKE)"/\]  
            TrainPy \--\> SNN\_Core  
            SNN\_Core \--\> MetaSNN  
        end  
        subgraph Inference \["推論 (Inference)"\]  
            Specialist \--\> Deployment\[/"snn\_research/deployment.py\<br/\>(SNNInferenceEngine)"/\]  
        end  
    end

    %% On-Demand Learning  
    subgraph OnDemand \["オンデマンド学習 (On-Demand Learning)"\]  
        direction TB  
        Agent\[/"run\_agent.py\<br/\>(AutonomousAgent)"/\]  
        Agent \--\>|未知のタスク| TrainPy  
        Agent \--\>|モデル検索| Registry\[/"snn\_research/distillation\<br/\>(ModelRegistry)"/\]  
    end  
      
    %% RL Loop  
    subgraph RL\_Loop \["生物学的強化学習 (Bio-RL)"\]  
        direction LR  
        RLAgent \--\> RL\_BioSNN\[/"snn\_research/bio\_models\<br/\>(BioSNN)"/\]  
        RL\_BioSNN \--\> Env\[/"snn\_research/rl\_env\<br/\>(SimpleEnvironment)"/\]  
        Env \--\>|Reward| RL\_BioSNN  
    end

    %% Node styles  
    style LifeForm fill:\#cde4ff,stroke:\#333,stroke-width:2px  
    style Planner fill:\#ffe4c4,stroke:\#333,stroke-width:1px  
    style RLAgent fill:\#ffe4c4,stroke:\#333,stroke-width:1px  
    style Agent fill:\#d4edda,stroke:\#333,stroke-width:2px  
    style TrainPy fill:\#f8d7da,stroke:\#333,stroke-width:2px  
    style SNN\_Core fill:\#fff2cd,stroke:\#333,stroke-width:2px

## **3\. 主要な実行スクリプト**

本プロジェクトは、利用者の目的に応じて複数の実行エントリーポイントを提供します。

| スクリプト | 役割 | ユースケース |
| :---- | :---- | :---- |
| run\_life\_form.py | **【推奨】デジタル生命体の起動** | AIに自律的に思考・学習させ、その活動を観察したい場合。 |
| run\_planner.py | **高次認知プランナーの操作** | 「要約して分析」のような複雑なタスクをAIに解決させたい場合。 |
| run\_agent.py | **自律エージェントの操作** | 「感情分析」のような単一のタスクを解決させたい場合（必要なら新規学習も行う）。 |
| run\_rl\_agent.py | **生物学的強化学習の実行** | AIが試行錯誤から学習する様子を観察したい場合。（バックプロパゲーション不使用） |
| run\_evolution.py | **自己進化サイクルの実行** | AIに自己のコード（アーキテクチャ）を改善させるメタなプロセスを試したい場合。 |
| app/main.py | **対話UIの起動** | 学習済みの専門家モデルとチャット形式で対話したい場合。 |
| train.py | **専門家モデルの手動学習** | 特定のタスクの専門家SNNを自分で学習させたい場合。（開発者向け） |
| train\_planner.py | **プランナーモデルの手動学習** | 計画立案能力を持つプランナーSNNを学習させたい場合。（開発者向け） |

## **4\. システムの実行方法**

### **ステップ1: 環境設定**

まず、必要なPythonライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: 基本操作 (ユースケース別)**

#### **A) デジタル生命体の自律ループを開始する (run\_life\_form.py)**

AIの自律的な思考と学習のループを開始します。AIは自身の「好奇心」レベルに基づき、新たな探求（思考または行動）を行ったり、自己の性能改善を試みたりします。

\# 10回の「意識サイクル」を実行  
python run\_life\_form.py \--cycles 10

#### **B) 生物学的強化学習を実行する (run\_rl\_agent.py)**

バックプロパゲーションを使わず、エージェントが報酬だけを頼りに簡単なパターンマッチングタスクを学習する様子を観察します。

python run\_rl\_agent.py \--episodes 500

#### **C) アーキテクチャレベルの自己進化を試す (run\_evolution.py)**

意図的に低い初期精度を与えることで、エージェントにモデルの表現力不足を認識させ、アーキテクチャ（d\_modelなど）を自律的に強化させます。

\# 精度0.4という厳しい状況を与え、smallモデルのアーキテクチャ改善を促す  
python run\_evolution.py \--task\_description "高難度タスク" \--initial\_accuracy 0.4 \--model\_config "configs/models/small.yaml"

#### **D) 新しいSpiking Transformerアーキテクチャで学習する (train.py)**

large.yamlで定義された新しいSpikingTransformerアーキテクチャでモデルを学習させます。

python train.py \\  
    \--model\_config configs/models/large.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.paradigm=gradient\_based" \\  
    \--override\_config "training.gradient\_based.type=standard"

## **5\. プロジェクト構造**

snn3/  
├── app/                  \# UIアプリケーションとDIコンテナ  
├── configs/                \# 設定ファイル (base, models/\*.yaml)  
├── doc/                    \# ドキュメント  
├── scripts/                \# データ準備やベンチマークなどの補助スクリプト  
├── snn\_research/           \# SNNコア研究開発コード  
│   ├── agent/            \# 各種エージェント (自律、自己進化、生命体、強化学習)  
│   ├── cognitive\_architecture/ \# 高次認知機能 (プランナー、物理評価器等)  
│   ├── core/             \# SNNモデル (BreakthroughSNN, SpikingTransformer)  
│   ├── learning\_rules/   \# 生物学的学習則 (STDPなど)  
│   ├── rl\_env/           \# 強化学習環境  
│   └── training/         \# Trainerと損失関数  
├── train.py                \# 勾配ベース学習の実行スクリプト  
├── run\_rl\_agent.py         \# 生物学的強化学習の実行スクリプト  
└── run\_\*.py                \# その他の機能実行スクリプト  
