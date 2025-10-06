# matsushibadenki/snn3/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 根本的なエラー修正:
# SpikingTransformerのアーキテクチャを全面的に修正。
# RNN的な時間ステップ処理とTransformer的なシーケンス処理の混在がエラーの根本原因であったため、
# Spiking Transformerの標準的な実装に修正。
# 修正後の動作:
# 1. 外部で`time_steps`のループを実行する。
# 2. 各時間ステップにおいて、Attentionブロックはシーケンス全体(B, N, C)を受け取って処理する。
# 3. 各ブロック内のニューロンは、時間ステップのループを通じて内部状態（膜電位）を更新する。
# これにより、すべてのテンソル形状の不整合が解消される。
# TypeError修正:
# nn.Sequential内でタプルを返すLIFニューロンを使用していたため、TypeErrorが発生していた。
# STAttenBlock内のFFNを手動でアンパックして実行するように修正。
# TypeError修正 (SimpleSNN):
# SimpleSNNが古いBioLIFNeuronを引数なしで呼び出していた問題を修正。
# 他のモデルと整合性を取るため、AdaptiveLIFNeuronを使用するように変更。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional  # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type, cast
import math
from omegaconf import DictConfig, OmegaConf
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# from snn_research.bio_models.lif_neuron import BioLIFNeuron as LIFNeuron # 古いLIFニューロンのインポートを削除
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️


# --- ニューロンモデル ---
class AdaptiveLIFNeuron(nn.Module):
    """
    適応的発火閾値を持つLIFニューロン (表現力向上のための標準ニューロン)
    """
    def __init__(
        self,
        features: int,
        tau: float = 2.0,
        base_threshold: float = 1.0,
        adaptation_strength: float = 0.1,
        target_spike_rate: float = 0.02,
    ):
        super().__init__()
        self.tau = tau
        self.base_threshold = base_threshold
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.surrogate_function = surrogate.ATan()
        self.mem_decay = math.exp(-1.0 / tau)
        self.register_buffer(
            "adaptive_threshold", torch.full((features,), base_threshold)
        )
        self.adaptive_threshold: torch.Tensor
        self.register_buffer("mem", torch.zeros(1, features))
        self.mem: torch.Tensor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 入力テンソルの次元数に応じて膜電位バッファの形状を調整
        if self.mem.shape[0] != x.shape[0] or self.mem.device != x.device or self.mem.ndim != x.ndim:
            self.mem = torch.zeros_like(x, device=x.device)

        mem_this_step = self.mem * self.mem_decay + x
        spike = self.surrogate_function(mem_this_step - self.adaptive_threshold)
        
        mem_for_next_step = mem_this_step * (1.0 - spike)
        self.mem = mem_for_next_step.detach()

        if self.training:
            with torch.no_grad():
                spike_rate_error = spike.mean() - self.target_spike_rate
                self.adaptive_threshold += self.adaptation_strength * spike_rate_error
                self.adaptive_threshold.clamp_(min=0.5)

        return spike, mem_this_step

class DendriticNeuron(nn.Module):
    """
    Phase 4: 樹状突起演算を模倣したニューロン。
    """
    def __init__(self, input_features: int, num_branches: int, branch_features: int):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_features, branch_features),
                AdaptiveLIFNeuron(branch_features)
            ) for _ in range(num_branches)
        ])
        self.soma_lif = AdaptiveLIFNeuron(branch_features * num_branches)
        self.output_projection = nn.Linear(branch_features * num_branches, input_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_outputs = [branch(x)[0] for branch in self.branches]
        concatenated_spikes = torch.cat(branch_outputs, dim=-1)
        soma_spike, soma_mem = self.soma_lif(concatenated_spikes)
        output = self.output_projection(soma_spike)
        return output, soma_mem

class SNNLayerNorm(nn.Module):
    """SNNのためのタイムステップごとに行うLayerNorm"""
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
    def forward(self, x):
        return self.norm(x)

# --- 予測符号化レイヤー ---
class PredictiveCodingLayer(nn.Module):
    """予測符号化を実行する単一の階層レイヤー。"""
    def __init__(self, d_model: int, d_state: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.generative_fc = nn.Linear(d_state, d_model)
        if neuron_class == DendriticNeuron:
            self.generative_neuron = neuron_class(input_features=d_model, **neuron_params)
        else:
            self.generative_neuron = neuron_class(features=d_model)

        self.inference_fc = nn.Linear(d_model, d_state)
        if neuron_class == DendriticNeuron:
            self.inference_neuron = neuron_class(input_features=d_state, **neuron_params)
        else:
            self.inference_neuron = neuron_class(features=d_state)
        
        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)

    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, _ = self.generative_neuron(self.generative_fc(top_down_state))
        prediction_error = F.relu(bottom_up_input - prediction)
        prediction_error = self.norm_error(prediction_error)
        state_update, inference_mem = self.inference_neuron(self.inference_fc(prediction_error))
        updated_state = self.norm_state(top_down_state + state_update)
        return updated_state, prediction_error, prediction, inference_mem

# --- コアSNNモデル (予測符号化) ---
class BreakthroughSNN(nn.Module):
    """リカレント予測符号化アーキテクチャを実装した階層的SNN"""
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        neuron_config = neuron_config or {"type": "lif"}
        neuron_type = neuron_config.get("type", "lif")
        neuron_class: Type[nn.Module]

        if neuron_type == "dendritic":
            neuron_class = DendriticNeuron
            neuron_params = {
                "num_branches": neuron_config.get("num_branches", 4),
                "branch_features": neuron_config.get("branch_features", d_model // 4)
            }
            self.input_encoder = nn.Sequential(nn.Linear(d_model, d_model), DendriticNeuron(input_features=d_model, **neuron_params))
            print("💡 BreakthroughSNN initialized with Dendritic Neurons.")
        else:
            neuron_class = AdaptiveLIFNeuron
            neuron_params = {}
            self.input_encoder = nn.Sequential(nn.Linear(d_model, d_model), AdaptiveLIFNeuron(features=d_model))
        
        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, n_head, neuron_class, neuron_params) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_spikes: bool = False,
        return_full_mems: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        states = [torch.zeros(batch_size, self.d_state, device=input_ids.device) for _ in range(self.num_layers)]
        
        total_spikes = torch.tensor(0.0, device=input_ids.device)
        total_mem_potential = torch.tensor(0.0, device=input_ids.device)
        all_hidden_states: List[torch.Tensor] = []
        all_mems_list: List[torch.Tensor] = []

        # spikingjellyの作法に従い、処理開始前にネットワークの状態をリセット
        functional.reset_net(self)

        for i in range(seq_len):
            embedded_token = token_emb[:, i, :]
            bottom_up_input, _ = self.input_encoder(embedded_token)
            
            layer_errors: List[torch.Tensor] = []
            layer_mems: List[torch.Tensor] = []
            
            for j in range(self.num_layers):
                states[j], error, _, mem = self.pc_layers[j](bottom_up_input, states[j])
                bottom_up_input = error
                layer_errors.append(error)
                layer_mems.append(mem)
            
            top_most_state = states[-1]
            _, _, final_prediction, _ = self.pc_layers[-1](
                torch.zeros_like(bottom_up_input, device=input_ids.device),
                top_most_state
            )
            all_hidden_states.append(final_prediction)

            if return_spikes or return_full_mems:
                total_spikes += sum(err.mean() for err in layer_errors)
                current_mem_avg = sum((m.abs().mean() for m in layer_mems), start=torch.tensor(0.0, device=input_ids.device))
                total_mem_potential += current_mem_avg
                if return_full_mems:
                    all_mems_list.append(current_mem_avg)

        final_hidden_states = torch.stack(all_hidden_states, dim=1)
        
        if output_hidden_states:
            final_output = final_hidden_states
        else:
            final_output = self.output_projection(final_hidden_states)

        avg_spikes = total_spikes / seq_len if return_spikes and seq_len > 0 else torch.tensor(0.0)
        final_mem = torch.stack(all_mems_list) if return_full_mems and all_mems_list else total_mem_potential / seq_len if seq_len > 0 else torch.tensor(0.0)

        return final_output, avg_spikes, final_mem

# --- ◾️◾️◾️↓Spiking Transformer アーキテクチャ (エラー修正済)↓◾️◾️◾️ ---
class SpikeDrivenSelfAttention(nn.Module):
    """スパイク駆動の自己注意機構。"""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.neuron_q = AdaptiveLIFNeuron(features=d_model)
        self.neuron_k = AdaptiveLIFNeuron(features=d_model)
        self.neuron_out = AdaptiveLIFNeuron(features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_spike, _ = self.neuron_q(q)
        k_spike, _ = self.neuron_k(k)

        q_spike_mha = q_spike.view(B, N, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k_spike_mha = k_spike.view(B, N, self.n_head, self.d_head).permute(0, 2, 3, 1)
        v_mha = v.view(B, N, self.n_head, self.d_head).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q_spike_mha, k_spike_mha) / math.sqrt(self.d_head)
        attn_weights_spike = torch.sigmoid(attn_scores)

        attn_output = torch.matmul(attn_weights_spike, v_mha)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        
        output = self.out_proj(attn_output)
        output_spike, _ = self.neuron_out(output)
        return output_spike

class STAttenBlock(nn.Module):
    """空間時間アテンションブロック。"""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = SpikeDrivenSelfAttention(d_model, n_head)
        self.norm2 = SNNLayerNorm(d_model)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # FFNを分解して、LIFニューロンのタプル出力を扱えるようにする
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.lif1 = AdaptiveLIFNeuron(features=d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.lif2 = AdaptiveLIFNeuron(features=d_model)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Sequence, Channels)
        x = x + self.attn(self.norm1(x))
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # FFNを手動で実行し、タプルをアンパックする
        identity = x
        out = self.norm2(x)
        out = self.fc1(out)
        out_spike, _ = self.lif1(out) # タプルをアンパック
        out = self.fc2(out_spike)
        out_spike, _ = self.lif2(out) # タプルをアンパック
        
        x = identity + out_spike
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️
        return x

class SpikingTransformer(nn.Module):
    """時間価値を最大化する、空間時間アテンションを備えたSpiking Transformer。"""
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model)) # 十分に大きいシーケンス長を確保
        
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        print(f"🚀 Spiking Transformer (STAtten) initialized with {num_layers} layers.")

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, return_full_mems: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x_embed = self.token_embedding(input_ids)
        x_embed = x_embed + self.pos_embedding[:, :seq_len, :]

        functional.reset_net(self)

        outputs_over_time = []
        spike_outputs_over_time = []

        for t in range(self.time_steps):
            x_t = x_embed
            for layer in self.layers:
                x_t = layer(x_t)
            
            # 最後のレイヤーの出力（スパイク）を記録
            spike_outputs_over_time.append(x_t)
            # 最終的なロジット計算のため、スパイクではない値（膜電位など）も保持したいが、
            # ここでは簡略化のためスパイクを積分（合計）する
            if len(outputs_over_time) > 0:
                outputs_over_time.append(outputs_over_time[-1] + x_t)
            else:
                outputs_over_time.append(x_t)
        
        # 時間方向に積分（合計）した値を最終出力とする
        final_output = outputs_over_time[-1]

        logits = self.output_projection(final_output)

        total_spikes = torch.tensor(0.0, device=x_embed.device)
        if return_spikes:
             # 全時間ステップ、全ニューロンのスパイクを合計
             for spikes in spike_outputs_over_time:
                 total_spikes += spikes.sum()
        
        # バッチとシーケンスで平均化
        avg_spikes = total_spikes / (self.time_steps * batch_size * seq_len) if return_spikes else torch.tensor(0.0)
        avg_mems = torch.tensor(0.0, device=x_embed.device)  # Placeholder

        return logits, avg_spikes, avg_mems

# --- ◾️◾️◾️↑Spiking Transformer アーキテクチャ↑◾️◾️◾️ ---

class SimpleSNN(nn.Module):
    """
    基本的なLIFニューロンで構成されたシンプルなSNNモデル。
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleSNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.lif1 = AdaptiveLIFNeuron(features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = AdaptiveLIFNeuron(features=output_size)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        T, B, _ = x.shape
        functional.reset_net(self)
        
        outputs = []
        for t in range(T):
            x_t = x[t, :, :]
            out = self.fc1(x_t)
            out, _ = self.lif1(out)
            out = self.fc2(out)
            out, _ = self.lif2(out)
            outputs.append(out)
            
        return torch.stack(outputs, dim=0)

class SNNCore(nn.Module):
    """
    設定に応じて適切なSNNアーキテクチャをインスタンス化するラッパークラス。
    """
    def __init__(self, config: DictConfig, vocab_size: int):
        super(SNNCore, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        # SNNCoreに渡されるconfigは既にモデル設定そのものであるため、.modelアクセスを削除
        model_type = self.config.get("architecture_type", self.config.get("type", "simple"))

        self.model: nn.Module

        # .modelアクセスを削除
        params_untyped = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(params_untyped, Dict):
            raise ValueError(f"Model configuration must be a dictionary. Got: {type(params_untyped)}")
        
        params: Dict[str, Any] = cast(Dict[str, Any], params_untyped)

        if model_type == "predictive_coding":
            self.model = BreakthroughSNN(
                vocab_size=vocab_size,
                d_model=params.get("d_model", 256),
                d_state=params.get("d_state", 128),
                num_layers=params.get("num_layers", 4),
                time_steps=params.get("time_steps", 20),
                n_head=params.get("n_head", 4),
                neuron_config=params.get("neuron", {})
            )
        elif model_type == "spiking_transformer":
            self.model = SpikingTransformer(
                vocab_size=vocab_size,
                d_model=params.get("d_model", 512),
                n_head=params.get("n_head", 8),
                num_layers=params.get("num_layers", 12),
                time_steps=params.get("time_steps", 32)
            )
        elif model_type == "simple":
            self.model = SimpleSNN(
                input_size=params.get("input_size", 10),
                hidden_size=params.get("hidden_size", 50),
                output_size=params.get("output_size", 10)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
