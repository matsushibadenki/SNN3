# matsushibadenki/snn3/SNN3-27170475db1dde34e4e83ac31427cebd290f9474/snn_research/agent/autonomous_agent.py
# ファイルパス: matsushibadenki/snn3/SNN3-176e5ceb739db651438b22d74c0021f222858011/snn_research/agent/autonomous_agent.py
# タイトル: 自律エージェント
# 機能説明: mypyエラーを解消するため、find_expert内の未定義変数へのアクセスを修正。

from typing import Dict, Any, Optional
import asyncio
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf

from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.tools.web_crawler import WebCrawler
from .memory import Memory as AgentMemory
from snn_research.deployment import SNNInferenceEngine


class AutonomousAgent:
    """
    自律的にタスクを実行するエージェントのベースクラス。
    """
    def __init__(self, name: str, planner: HierarchicalPlanner, model_registry: ModelRegistry, memory: AgentMemory, web_crawler: WebCrawler, accuracy_threshold: float = 0.6, energy_budget: float = 10000.0):
        self.name = name
        self.planner = planner
        self.model_registry = model_registry
        self.memory = memory
        self.web_crawler = web_crawler
        self.current_state = {"agent_name": name} # 初期状態
        self.accuracy_threshold = accuracy_threshold
        self.energy_budget = energy_budget

    def execute(self, task_description: str) -> str:
        """
        与えられたタスクを実行する。
        """
        print(f"Agent '{self.name}' is executing task: {task_description}")

        if "research" in task_description or "latest information" in task_description:
            return self.learn_from_web(task_description)

        expert = asyncio.run(self.find_expert(task_description))
        action = "execute_task_with_expert" if expert else "execute_task_general"
        expert_id = [expert['model_id']] if expert else []

        if expert:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using expert model '{expert['model_id']}'."
        else:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using general capabilities (no specific expert found)."
        
        self.memory.record_experience(
            state=self.current_state,
            action=action,
            result={"status": "SUCCESS", "details": result},
            reward=1.0,
            expert_used=expert_id,
            decision_context={"reason": "Direct execution command received."}
        )
        return result

    async def find_expert(self, task_description: str) -> Dict[str, Any] | None:
        """
        タスクに最適な専門家モデルをモデルレジストリから検索する。
        """
        safe_task_description = task_description.lower().replace(" ", "_")
        candidate_experts = await self.model_registry.find_models_for_task(safe_task_description, top_k=5)
        
        if not candidate_experts:
            print(f"最適な専門家が見つかりませんでした: {safe_task_description}")
            return None

        # 精度とエネルギーの条件を満たすモデルを探す
        for expert in candidate_experts:
            metrics = expert.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            spikes = metrics.get("avg_spikes_per_sample", float('inf'))
            if accuracy >= self.accuracy_threshold and spikes <= self.energy_budget:
                print(f"✅ 条件を満たす専門家を発見しました: {expert['model_id']} (Accuracy: {accuracy:.4f}, Spikes: {spikes:.2f})")
                return expert

        # 妥協案: 条件を満たすモデルがなくても、最も精度の高いモデルを返す
        print(f"⚠️ 専門家は見つかりましたが、精度/エネルギー要件を満たすモデルがありませんでした。")
        best_candidate = candidate_experts[0]
        print(f"   - 最高性能モデル: {best_candidate.get('metrics', {})} (要件: accuracy >= {self.accuracy_threshold}, spikes <= {self.energy_budget})")
        print(f"   - 妥協案として、最高性能モデル '{best_candidate.get('model_id')}' を採用します。")
        return best_candidate

    def learn_from_web(self, topic: str) -> str:
        """
        Webクローラーを使って情報を収集し、知識を更新する。
        """
        print(f"Agent '{self.name}' is learning about '{topic}' from the web.")
        urls = self._search_for_urls(topic)
        task_name = f"learn_from_web: {topic}"
        if not urls:
            result_details = "Could not find relevant information on the web."
            self.memory.record_experience(
                state=self.current_state, action=task_name,
                result={"status": "FAILURE", "details": result_details},
                reward=-1.0, expert_used=["web_crawler"],
                decision_context={"reason": "No relevant URLs found."}
            )
            return result_details

        content = self.web_crawler.crawl(urls[0])
        summary = self._summarize(content)

        self.memory.record_experience(
            state=self.current_state, action=task_name,
            result={"status": "SUCCESS", "summary": summary},
            reward=1.0, expert_used=["web_crawler", "summarizer"],
            decision_context={"reason": "Information successfully retrieved and summarized."}
        )
        return f"Successfully learned about '{topic}'. Summary: {summary}"

    def _search_for_urls(self, query: str) -> list[str]:
        # ToDo: 実際の検索エンジンAPIに置き換える
        return [f"https://www.google.com/search?q={query.replace(' ', '+')}"]

    def _summarize(self, text: str) -> str:
        # ToDo: より高度な要約モデルに置き換える
        return text[:150] + "..."
    
    async def handle_task(self, task_description: str, unlabeled_data_path: Optional[str] = None, force_retrain: bool = False) -> Optional[Dict[str, Any]]:
        """
        タスクを処理する中心的なメソッド。専門家を検索し、いなければ学習を試みる。
        """
        print(f"--- Handling Task: {task_description} ---")
        self.memory.record_experience(self.current_state, "handle_task", {"task": task_description}, 0.0, [], {"reason": "Task received"})

        expert_model = await self.find_expert(task_description)

        if expert_model and not force_retrain:
            print(f"✅ Found existing expert model: {expert_model['model_id']}")
            return expert_model
        
        if unlabeled_data_path:
            print("- No suitable expert found or retraining forced. Initiating on-demand learning...")
            try:
                from app.containers import TrainingContainer
                container = TrainingContainer()
                container.config.from_yaml("configs/base_config.yaml")
                container.config.from_yaml("configs/models/small.yaml")

                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                # 依存関係を正しい順序で構築し、Optimizerにパラメータを渡す
                device = container.device()
                student_model = container.snn_model().to(device)
                optimizer = container.optimizer(params=student_model.parameters())
                scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

                distillation_trainer = container.distillation_trainer(
                    model=student_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device
                )

                manager = KnowledgeDistillationManager(
                    student_model=student_model,
                    trainer=distillation_trainer,
                    teacher_model_name=container.config.training.gradient_based.distillation.teacher_model(),
                    tokenizer_name=container.config.data.tokenizer_name(),
                    model_registry=self.model_registry,
                    device=device
                )
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                
                with open(unlabeled_data_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
                
                train_loader = manager.prepare_dataset(texts, max_length=container.config.model.time_steps(), batch_size=container.config.training.batch_size())
                
                new_model_info = await manager.run_distillation(
                    train_loader=train_loader,
                    val_loader=train_loader,
                    epochs=3,
                    model_id=task_description,
                    task_description=f"Expert for {task_description}",
                    student_config=container.config.model.to_dict()
                )
                self.memory.record_experience(self.current_state, "on_demand_learning", new_model_info, 1.0, [new_model_info['model_id']], {"reason": "New expert created"})
                return new_model_info

            except Exception as e:
                print(f"❌ On-demand learning failed: {e}")
                self.memory.record_experience(self.current_state, "on_demand_learning", {"error": str(e)}, -1.0, [], {"reason": "Training failed"})
                return None
        
        # データがない場合でも、妥協案として見つけたモデルを返す
        if expert_model:
            return expert_model
            
        print("- No expert found and no data provided for training.")
        self.memory.record_experience(self.current_state, "handle_task", {"status": "failed"}, -1.0, [], {"reason": "No expert and no data"})
        return None

    async def run_inference(self, model_info: Dict[str, Any], prompt: str) -> None:
        """
        指定されたモデルで推論を実行する。
        """
        print(f"Running inference with model {model_info.get('model_id', 'N/A')} on prompt: {prompt}")

        model_config = model_info.get('config')
        
        if not model_config:
            print("⚠️ Warning: Model config not found in registry. Falling back to default 'small' model config.")
            try:
                small_config_path = next(Path('.').rglob('configs/models/small.yaml'))
                model_config = OmegaConf.load(small_config_path).get('model', {})
            except (StopIteration, FileNotFoundError):
                print("❌ Error: Default 'small.yaml' not found. Cannot proceed with inference.")
                return

        deployment_config = {
            'deployment': {
                'model_path': model_info.get('model_path'),
                'tokenizer_path': "gpt2",
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'model': model_config
        }
        config = OmegaConf.create(deployment_config)

        try:
            inference_engine = SNNInferenceEngine(config=config)
            
            full_response = ""
            print("Response: ", end="", flush=True)
            for chunk in inference_engine.generate(prompt, max_len=50):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n--- Inference Complete ---")
            
            self.memory.record_experience(self.current_state, "inference", {"prompt": prompt, "response": full_response}, 0.5, [model_info.get('model_id')], {})

        except Exception as e:
            print(f"\n❌ Inference failed: {e}")
            self.memory.record_experience(self.current_state, "inference", {"error": str(e)}, -0.5, [model_info.get('model_id')], {})
