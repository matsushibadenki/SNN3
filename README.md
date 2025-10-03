# **Project SNN: A Predictive Digital Life Form (v4.1)**

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

### **A) データ読み込みと学習**

SNNモデルに知性を与えるための、様々な学習方法を紹介します。

#### **A-1. 手動でのモデル学習**

研究者が意図した設定でモデルを学習させる、最も基本的な方法です。アーキテクチャや学習方法を自由に組み合わせて実験できます。

例：新しいSpiking Transformerアーキテクチャでモデルを学習させる  
large.yamlで定義された大規模なSpiking Transformerモデルを、gradient\_based（勾配ベース）という標準的な方法で学習させます。  
python snn-cli.py train \\  
    \--model\_config configs/models/large.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.paradigm=gradient\_based" \\  
    \--override\_config "training.gradient\_based.type=standard"

#### **A-2. AIによるオンデマンド学習**

AI自身が必要だと判断した際に、自律的に新しい能力（専門家モデル）を獲得させる方法です。

例：「文章要約」の専門家モデルをAIに自動で学習させる  
エージェントに「文章要約」タスクを依頼します。もし対応する専門家モデルが存在しない場合、エージェントは提供されたデータ (--unlabeled\_data\_path) を使い、大規模言語モデルから知識を蒸留して、新しい専門家を自動で育成します。  
python snn-cli.py agent solve \\  
    \--task "文章要約" \\  
    \--unlabeled\_data\_path data/sample\_data.jsonl

#### **A-3. 行動を通じた学習（強化学習）**

バックプロパゲーション（勾配計算）を一切使わず、生物のように環境との試行錯誤（行動と報酬）から直接スキルを学習させる、全く異なるパラダイムです。

例：強化学習エージェントにパターンマッチングを学習させる  
エージェントは、環境から与えられた目標パターンと自身が出力したパターンを比較し、「報酬」を頼りに、正解のパターンを自力で見つけ出すように学習します。  
python snn-cli.py rl run \--episodes 100

### **B) 推論**

学習済みのモデルを使って、実際にタスクを実行したり、対話したりする方法です。

#### **B-1. 対話UIによる推論**

学習済みの専門家モデルとチャット形式で対話するためのWeb UIを起動します。最も手軽にAIの能力を体験できる方法です。

**例：標準モデル（small）と対話する**

python app/main.py

例：中規模モデル（medium）と対話する  
\--model\_config を変更することで、対話するモデルを切り替えられます。  
python app/main.py \--model\_config configs/models/medium.yaml

#### **B-2. CLIによる単一タスクの推論**

コマンドラインから直接、特定のタスクを実行させます。

例：「感情分析」タスクを実行する  
学習済みの感情分析モデルを呼び出し、与えられたプロンプトがポジティブかネガティブかを推論させます。  
python snn-cli.py agent solve \\  
    \--task "感情分析" \\  
    \--prompt "この映画は本当に素晴らしかった！"

#### **B-3. 複雑なタスクの推論（プランナー）**

複数のスキルを組み合わせる必要がある複雑な要求を、AIに計画させて実行させます。

例：「要約」と「感情分析」を組み合わせたタスクを実行する  
AIはタスク要求を理解し、「まず文章を要約し、次にその要約文の感情を分析する」という計画を自ら立て、2つの専門家モデルを順番に呼び出してタスクを遂行します。  
python snn-cli.py planner execute \\  
    \--request "この記事を要約して、その内容の感情を分析してください。" \\  
    \--context "SNNは非常にエネルギー効率が高いことで知られているが、その性能はまだANNに及ばない点もある。"

#### **B-4. 自律的思考ループ（デジタル生命体）**

AIに特定のタスクを与えるのではなく、自らの「好奇心」に基づき、何をすべきかを自律的に考えさせ、行動させる、最も高度な実行モードです。

例：デジタル生命体の「意識」ループを開始する  
AIは自身の内部状態を観測し、「退屈している」と感じれば、新しい知識を探求したり（プランナーや強化学習の実行）、自己の性能が低いと感じれば、自身のアーキテクチャを改善（自己進化）したりします。  
python snn-cli.py life-form start \--cycles 10

## **4\. プロジェクト構造**

snn3/

├── app/ \# UIアプリケーションとDIコンテナ

├── configs/ \# 設定ファイル (base, models/.yaml)

├── doc/ \# ドキュメント

├── scripts/ \# データ準備やベンチマークなどの補助スクリプト

├── snn\_research/ \# SNNコア研究開発コード

│ ├── agent/ \# 各種エージェント (自律、自己進化、生命体、強化学習)

│ ├── cognitive\_architecture/ \# 高次認知機能 (プランナー、物理評価器等)

│ ├── core/ \# SNNモデル (BreakthroughSNN, SpikingTransformer)

│ ├── learning\_rules/ \# 生物学的学習則 (STDPなど)

│ ├── rl\_env/ \# 強化学習環境

│ └── training/ \# Trainerと損失関数

├── snn-cli.py \# ✨ 新規: 統合CLIツール

├── train.py \# 勾配ベース学習の実行スクリプト (CLIから呼び出される)

└── ... (その他のrun\_.pyスクリプトは内部的に利用、または将来的に廃止)
