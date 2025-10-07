# matsushibadenki/snn3/app/containers.py
# DIコンテナの定義ファイル (完全版)
#
# 機能:
# - 勾配ベース学習と生物学的学習を含む複数の学習パラダイムをDIコンテナで管理。
# - 設定ファイルの `training.paradigm` の値に応じて、適切なコンポーネント群を構築する。
# - 既存の全機能を維持しつつ、新しい学習方法への拡張性を確保。
# - 変更点: SpikingTransformerを新しいアーキテクチャとして追加し、設定で切り替えられるように修正。
# - 変更点: 生物学的強化学習(BioRLTrainer)関連のプロバイダを再統合し、完全な状態に復元。
# - mypyエラー修正: BioSNNのインポートパスを修正。
# - BugFix: SNNCoreに設定ファイルの`model`セクションを正しく渡すように修正。

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- プロジェクト内モジュールのインポート ---
from snn_research.core.snn_core import SNNCore, BreakthroughSNN, SpikingTransformer
from snn_research.deployment import SNNInferenceEngine
from snn_research.training.losses import CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss
from snn_research.training.trainers import BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer, PhysicsInformedTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter
from snn_research.distillation.model_registry import SimpleModelRegistry
import redis
from snn_research.tools.web_crawler import WebCrawler

# --- 生物学的学習のためのインポート ---
from snn_research.learning_rules.stdp import STDP
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP
from snn_research.bio_models.simple_network import BioSNN
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.simple_env import SimpleEnvironment
from snn_research.training.bio_trainer import BioRLTrainer

from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.agent.memory import Memory


def get_auto_device() -> str:
    """実行環境に最適なデバイスを自動的に選択する。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _calculate_t_max(epochs: int, warmup_epochs: int) -> int:
    """学習率スケジューラのT_maxを計算する"""
    return max(1, epochs - warmup_epochs)

def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    """ウォームアップ付きのCosineAnnealingスケジューラを生成するファクトリ関数。"""
    warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    main_scheduler_t_max = _calculate_t_max(epochs=epochs, warmup_epochs=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=main_scheduler_t_max)
    return SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])


class TrainingContainer(containers.DeclarativeContainer):
    """学習に関連するオブジェクトの依存関係を管理するコンテナ。"""
    config = providers.Configuration()

    # --- 共通ツール ---
    device = providers.Factory(get_auto_device)

    # --- 共通コンポーネント ---
    tokenizer = providers.Factory(AutoTokenizer.from_pretrained, pretrained_model_name_or_path=config.data.tokenizer_name)

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # --- アーキテクチャ選択 ---
    # SNNCoreに設定ファイルの`model`セクションを正しく渡すように修正。
    snn_model = providers.Factory(
        SNNCore,
        config=config.model,
        vocab_size=tokenizer.provided.vocab_size,
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    meta_cognitive_snn = providers.Factory(
        MetaCognitiveSNN,
        **(config.training.meta_cognition.to_dict() or {})
    )

    # === 勾配ベース学習 (gradient_based) のためのプロバイダ ===
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(_create_scheduler, optimizer=optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)

    standard_loss = providers.Factory(
        CombinedLoss,
        tokenizer=tokenizer,
        ce_weight=config.training.gradient_based.loss.ce_weight,
        spike_reg_weight=config.training.gradient_based.loss.spike_reg_weight,
        sparsity_reg_weight=config.training.gradient_based.loss.sparsity_reg_weight,
        mem_reg_weight=config.training.gradient_based.loss.mem_reg_weight,
    )
    distillation_loss = providers.Factory(
        DistillationLoss,
        tokenizer=tokenizer,
        ce_weight=config.training.gradient_based.distillation.loss.ce_weight,
        distill_weight=config.training.gradient_based.distillation.loss.distill_weight,
        spike_reg_weight=config.training.gradient_based.distillation.loss.spike_reg_weight,
        sparsity_reg_weight=config.training.gradient_based.distillation.loss.sparsity_reg_weight,
        mem_reg_weight=config.training.gradient_based.distillation.loss.mem_reg_weight,
        temperature=config.training.gradient_based.distillation.loss.temperature,
    )

    teacher_model = providers.Factory(AutoModelForCausalLM.from_pretrained, pretrained_model_name_or_path=config.training.gradient_based.distillation.teacher_model)
    standard_trainer = providers.Factory(
        BreakthroughTrainer, model=snn_model, optimizer=optimizer, criterion=standard_loss, scheduler=scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.gradient_based.grad_clip_norm,
        rank=-1, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn,
    )
    distillation_trainer = providers.Factory(
        DistillationTrainer, model=snn_model, optimizer=optimizer, criterion=distillation_loss, scheduler=scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.gradient_based.grad_clip_norm,
        rank=-1, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn,
    )

    # === 自己教師あり学習 (self_supervised) のためのプロバイダ ===
    ssl_optimizer = providers.Factory(AdamW, lr=config.training.self_supervised.learning_rate)
    ssl_scheduler = providers.Factory(_create_scheduler, optimizer=ssl_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.self_supervised.warmup_epochs)

    self_supervised_loss = providers.Factory(
        SelfSupervisedLoss,
        tokenizer=tokenizer,
        prediction_weight=config.training.self_supervised.loss.prediction_weight,
        spike_reg_weight=config.training.self_supervised.loss.spike_reg_weight,
        sparsity_reg_weight=config.training.self_supervised.loss.sparsity_reg_weight,
        mem_reg_weight=config.training.self_supervised.loss.mem_reg_weight,
    )

    self_supervised_trainer = providers.Factory(
        SelfSupervisedTrainer, model=snn_model, optimizer=ssl_optimizer, criterion=self_supervised_loss, scheduler=ssl_scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.self_supervised.grad_clip_norm,
        rank=-1, use_amp=config.training.self_supervised.use_amp, log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn,
    )

    # === 物理情報学習 (physics_informed) のためのプロバイダ ===
    pi_optimizer = providers.Factory(AdamW, lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    
    physics_informed_loss = providers.Factory(
        PhysicsInformedLoss,
        tokenizer=tokenizer,
        ce_weight=config.training.physics_informed.loss.ce_weight,
        spike_reg_weight=config.training.physics_informed.loss.spike_reg_weight,
        mem_smoothness_weight=config.training.physics_informed.loss.mem_smoothness_weight,
    )

    physics_informed_trainer = providers.Factory(
        PhysicsInformedTrainer, model=snn_model, optimizer=pi_optimizer, criterion=physics_informed_loss, scheduler=pi_scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.physics_informed.grad_clip_norm,
        rank=-1, use_amp=config.training.physics_informed.use_amp, log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn,
    )

    # === 生物学的学習 (biologically_plausible) のためのプロバイダ ===
    bio_learning_rule = providers.Selector(
        config.training.biologically_plausible.learning_rule,
        STDP=providers.Factory(
            STDP,
            learning_rate=config.training.biologically_plausible.stdp.learning_rate,
            a_plus=config.training.biologically_plausible.stdp.a_plus,
            a_minus=config.training.biologically_plausible.stdp.a_minus,
            tau_trace=config.training.biologically_plausible.stdp.tau_trace,
        ),
        REWARD_MODULATED_STDP=providers.Factory(
            RewardModulatedSTDP,
            learning_rate=config.training.biologically_plausible.reward_modulated_stdp.learning_rate,
            tau_eligibility=config.training.biologically_plausible.reward_modulated_stdp.tau_eligibility,
            # STDPの設定を継承して利用
            a_plus=config.training.biologically_plausible.stdp.a_plus,
            a_minus=config.training.biologically_plausible.stdp.a_minus,
            tau_trace=config.training.biologically_plausible.stdp.tau_trace,
        ),
    )

    bio_snn_model = providers.Factory(
        BioSNN,
        # 仮の値を設定。実際の利用シーンに応じて設定を見直す必要があります。
        n_input=10,
        n_hidden=50,
        n_output=2,
        neuron_params=config.training.biologically_plausible.neuron,
        learning_rule=bio_learning_rule,
    )

    rl_environment = providers.Factory(SimpleEnvironment, pattern_size=10)

    rl_agent = providers.Factory(
        ReinforcementLearnerAgent,
        input_size=10,
        output_size=10,
        device=providers.Factory(get_auto_device),
    )

    bio_rl_trainer = providers.Factory(
        BioRLTrainer,
        agent=rl_agent,
        env=rl_environment,
    )

    # === 学習可能プランナー (PlannerSNN) のためのプロバイダ ===
    planner_snn = providers.Factory(
        PlannerSNN, vocab_size=tokenizer.provided.vocab_size, d_model=config.model.d_model,
        d_state=config.model.d_state, num_layers=config.model.num_layers,
        time_steps=config.model.time_steps, n_head=config.model.n_head,
        num_skills=10 # 仮のスキル数
    )
    planner_optimizer = providers.Factory(AdamW, lr=config.training.planner.learning_rate)
    planner_loss = providers.Factory(PlannerLoss)

    # Redisクライアントのプロバイダ
    redis_client = providers.Singleton(
        redis.Redis,
        host=config.model_registry.redis.host,
        port=config.model_registry.redis.port,
        db=config.model_registry.redis.db,
        decode_responses=True,
    )

    # ModelRegistryのプロバイダ
    # Selectorでエラーが発生するため、現状の実装に合わせてFileベースのものを直接指定する
    model_registry = providers.Singleton(
        SimpleModelRegistry,
        registry_path=config.model_registry.file.path
    )


class AgentContainer(containers.DeclarativeContainer):
    """エージェントとプランナーの実行に必要な依存関係を管理するコンテナ。"""
    config = providers.Configuration()
    training_container = providers.Container(TrainingContainer, config=config)

    # --- 共通ツール ---
    device = providers.Factory(get_auto_device)
    model_registry = training_container.model_registry
    web_crawler = providers.Singleton(WebCrawler)
    rag_system = providers.Singleton(RAGSystem, vector_store_path=config.training.log_dir.concat("/vector_store"))
    memory = providers.Singleton(Memory, memory_path=config.training.log_dir.concat("/agent_memory.jsonl"))

    # --- 学習済みプランナーモデルのプロバイダ ---
    # train_planner.pyで学習させたモデルをロードする
    trained_planner_snn = providers.Factory(
        training_container.planner_snn
    )

    @providers.Singleton
    def loaded_planner_snn(trained_planner_snn, config, device):
        model_path = config.training.planner.model_path
        model = trained_planner_snn
        if os.path.exists(model_path):
            try:
                # state_dictの 'model_state_dict' キーをチェック
                checkpoint = torch.load(model_path, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(state_dict)
                print(f"✅ 学習済みPlannerSNNモデルを '{model_path}' から正常にロードしました。")
            except Exception as e:
                print(f"⚠️ PlannerSNNモデルのロードに失敗しました: {e}。未学習のモデルを使用します。")
        else:
            print(f"⚠️ PlannerSNNモデルが見つかりません: {model_path}。未学習のモデルを使用します。")
        return model.to(device)

    # --- プランナー ---
    hierarchical_planner = providers.Singleton(
        HierarchicalPlanner,
        model_registry=model_registry,
        rag_system=rag_system,
        planner_model=loaded_planner_snn,
        tokenizer_name=config.data.tokenizer_name,
        device=device,
    )

class AppContainer(containers.DeclarativeContainer):
    """GradioアプリやAPIなど、アプリケーション層の依存関係を管理するコンテナ。"""
    config = providers.Configuration()
    # Tools
    web_crawler = providers.Singleton(WebCrawler)
    device = providers.Factory(lambda cfg_device: get_auto_device() if cfg_device == "auto" else cfg_device, cfg_device=config.device)
    snn_inference_engine = providers.Singleton(SNNInferenceEngine, config=config)
    chat_service = providers.Factory(ChatService, snn_engine=snn_inference_engine, max_len=config.app.max_len)
    langchain_adapter = providers.Factory(SNNLangChainAdapter, snn_engine=snn_inference_engine)
