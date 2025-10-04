# matsushibadenki/snn3/app/containers.py
# DIコンテナの定義ファイル (完全版)
#
# 機能:
# - 勾配ベース学習と生物学的学習の2つのパラダイムをDIコンテナで管理。
# - 設定ファイルの `training.paradigm` の値に応じて、適切なコンポーネント群を構築する。
# - 既存の全機能を維持しつつ、新しい学習方法への拡張性を確保。
# - 変更点: SpikingTransformerを新しいアーキテクチャとして追加し、設定で切り替えられるように修正。
# - 変更点: 不要になった古い生物学的学習(BioTrainer)関連のプロバイダを削除。

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- プロジェクト内モジュールのインポート (既存) ---
from snn_research.core.snn_core import BreakthroughSNN, SpikingTransformer
from snn_research.deployment import SNNInferenceEngine
from snn_research.training.losses import CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss
from snn_research.training.trainers import BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer, PhysicsInformedTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter

from snn_research.distillation.model_registry import FileModelRegistry, RedisModelRegistry, ModelRegistry
import redis

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

    # --- 共通コンポーネント ---
    tokenizer = providers.Factory(AutoTokenizer.from_pretrained, pretrained_model_name_or_path=config.data.tokenizer_name)

    # --- アーキテクチャ選択 ---
    breakthrough_snn = providers.Factory(
        BreakthroughSNN, vocab_size=tokenizer.provided.vocab_size, d_model=config.model.d_model,
        d_state=config.model.d_state, num_layers=config.model.num_layers, time_steps=config.model.time_steps,
        n_head=config.model.n_head, neuron_config=config.model.neuron,
    )
    spiking_transformer = providers.Factory(
        SpikingTransformer, vocab_size=tokenizer.provided.vocab_size, d_model=config.model.d_model,
        n_head=config.model.n_head, num_layers=config.model.num_layers, time_steps=config.model.time_steps
    )
    snn_model = providers.Selector(
        config.model.architecture_type,
        predictive_coding=breakthrough_snn,
        spiking_transformer=spiking_transformer,
    )

    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    meta_cognitive_snn = providers.Factory(
        MetaCognitiveSNN,
        snn_model=snn_model,
        # configセクションが存在しない場合にNoneになるのを防ぐため、デフォルトの空辞書を指定
        **(config.training.meta_cognition.to_dict() or {})
    )

    # === 勾配ベース学習 (gradient_based) のためのプロバイダ ===
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(_create_scheduler, optimizer=optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
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
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

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

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    self_supervised_loss = providers.Factory(
        SelfSupervisedLoss,
        tokenizer=tokenizer,
        prediction_weight=config.training.self_supervised.loss.prediction_weight,
        spike_reg_weight=config.training.self_supervised.loss.spike_reg_weight,
        sparsity_reg_weight=config.training.self_supervised.loss.sparsity_reg_weight,
        mem_reg_weight=config.training.self_supervised.loss.mem_reg_weight,
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    self_supervised_trainer = providers.Factory(
        SelfSupervisedTrainer, model=snn_model, optimizer=ssl_optimizer, criterion=self_supervised_loss, scheduler=ssl_scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.self_supervised.grad_clip_norm,
        rank=-1, use_amp=config.training.self_supervised.use_amp, log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn,
    )

    # === 物理情報学習 (physics_informed) のためのプロバイダ ===
    pi_optimizer = providers.Factory(AdamW, lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    physics_informed_loss = providers.Factory(
        PhysicsInformedLoss,
        tokenizer=tokenizer,
        ce_weight=config.training.physics_informed.loss.ce_weight,
        spike_reg_weight=config.training.physics_informed.loss.spike_reg_weight,
        mem_smoothness_weight=config.training.physics_informed.loss.mem_smoothness_weight,
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    physics_informed_trainer = providers.Factory(
        PhysicsInformedTrainer, model=snn_model, optimizer=pi_optimizer, criterion=physics_informed_loss, scheduler=pi_scheduler,
        device=providers.Factory(get_auto_device), grad_clip_norm=config.training.physics_informed.grad_clip_norm,
        rank=-1, use_amp=config.training.physics_informed.use_amp, log_dir=config.training.log_dir,
        astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn,
    )
    
    # === 学習可能プランナー (PlannerSNN) のためのプロバイダ ===
    planner_snn = providers.Factory(
        PlannerSNN, vocab_size=tokenizer.provided.vocab_size, d_model=config.model.d_model,
        d_state=config.model.d_state, num_layers=config.model.num_layers,
        time_steps=config.model.time_steps, n_head=config.model.n_head
    )
    planner_optimizer = providers.Factory(AdamW, lr=config.training.planner.learning_rate)
    planner_loss = providers.Factory(PlannerLoss)

    # Redisクライアントのプロバイダ
    redis_client = providers.Singleton(
        redis.Redis,
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=True,
    )

    # ModelRegistryのプロバイダ
    model_registry = providers.Selector(
        config.model_registry.provider,
        file=providers.Singleton(FileModelRegistry, registry_path=config.model_registry.file.path),
        redis=providers.Singleton(RedisModelRegistry, redis_client=redis_client),
    )


class AppContainer(containers.DeclarativeContainer):
    """GradioアプリやAPIなど、アプリケーション層の依存関係を管理するコンテナ。"""
    config = providers.Configuration()
    # Tools
    web_crawler = providers.Singleton(WebCrawler)
    device = providers.Factory(lambda cfg_device: get_auto_device() if cfg_device == "auto" else cfg_device, cfg_device=config.inference.device)
    snn_inference_engine = providers.Singleton(SNNInferenceEngine, model_path=config.model.path, device=device)
    chat_service = providers.Factory(ChatService, snn_engine=snn_inference_engine, max_len=config.inference.max_len)
    langchain_adapter = providers.Factory(SNNLangChainAdapter, snn_engine=snn_inference_engine)

