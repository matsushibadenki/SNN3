# matsushibadenki/snn3/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
# 変更点:
# - 「時間」の価値を最大化するため、Spiking Transformerアーキテクチャを追加。
# - SpikeDrivenSelfAttention: スパイクベースの効率的な自己注意機構。
# - STAttenBlock: 空間と時間の両方を考慮するアテンションブロック。
# - SpikingTransformer: 新しい最先端モデルとして、STAttenBlockを統合。
# - mypyエラー修正: SpikingTransformer.forwardの戻り値の型をtorch.Tensorに統一。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional  # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type
import math
from omegaconf import DictConfig
from snn_research.bio_models.lif_neuron import BioLIFNeuron as LIFNeuron
from snn_research.models.spiking_transformer import SpikingTransformer


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
        if self.mem.shape[0] != x.shape[0] or self.mem.device != x.device:
            self.mem = torch.zeros(x.shape[0], x.shape[-1], device=x.device)

        self.mem = self.mem * self.mem_decay + x
        spike = self.surrogate_function(self.mem - self.adaptive_threshold)
        self.mem = self.mem * (1.0 - spike.detach())

        if self.training:
            with torch.no_grad():
                spike_rate_error = spike.mean() - self.target_spike_rate
                self.adaptive_threshold += self.adaptation_strength * spike_rate_error
                self.adaptive_threshold.clamp_(min=0.5)

        return spike, self.mem

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

        for i in range(seq_len):
            embedded_token = token_emb[:, i, :]
            bottom_up_input, _ = self.input_encoder(embedded_token)
            
            layer_errors: List[torch.Tensor] = []
            layer_mems: List[torch.Tensor] = []
            
            for j in range(self.num_layers):
                states[j], error, _, mem = self.pc_layers[j](bottom_up_input, states[j])
                # 次の層への入力は、現在の層の予測誤差（error）とする
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

# --- ◾️◾️◾️↓Spiking Transformer アーキテクチャ (mypyエラー修正済)↓◾️◾️◾️ ---
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
        self.neuron = AdaptiveLIFNeuron(features=d_model)

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        T, B, N, C = x_spike.shape  # (Time, Batch, Sequence, Channels)
        
        q = self.q_proj(x_spike).reshape(T, B, N, self.n_head, self.d_head)
        k = self.k_proj(x_spike).reshape(T, B, N, self.n_head, self.d_head)
        v = self.v_proj(x_spike).reshape(T, B, N, self.n_head, self.d_head)
        
        # 行列乗算の代わりに加算ベースの注意を計算 (簡略版)
        attn = torch.einsum('tbnhd,tbnhd->tbnh', q, k)
        
        # Softmaxの排除
        attn_weights_spike = (attn > 0.5).float() # 閾値でスパイク化
        
        # 加算ベースのValue適用
        attn_output = torch.einsum('tbnh,tbnhd->tbnhd', attn_weights_spike, v)
        
        output = self.out_proj(attn_output.reshape(T, B, N, C))
        output_spike, _ = self.neuron(output)
        
        return output_spike

class STAttenBlock(nn.Module):
    """空間時間アテンションブロック。"""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = SpikeDrivenSelfAttention(d_model, n_head)
        self.norm2 = SNNLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            AdaptiveLIFNeuron(features=d_model * 4),
            nn.Linear(d_model * 4, d_model),
        )
        self.neuron = AdaptiveLIFNeuron(features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Time, Batch, Sequence, Channels)
        # 空間アテンション
        x = x + self.attn(self.norm1(x))
        
        # 時間アテンション (簡略化のためFFN内で時間情報をミックス)
        ffn_out, _ = self.neuron(self.ffn(self.norm2(x)))
        x = x + ffn_out
        
        return x

class SpikingTransformer(nn.Module):
    """時間価値を最大化する、空間時間アテンションを備えたSpiking Transformer。"""
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, time_steps, d_model))
        
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        print(f"🚀 Spiking Transformer (STAtten) initialized with {num_layers} layers.")

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, return_full_mems: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # [Batch, Seq, Dim] -> [Time, Batch, Seq, Dim]
        # SpikingJellyに準拠し、時間軸を先頭に
        x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        total_spikes = torch.tensor(0.0, device=x.device)
        
        for layer in self.layers:
            x = layer(x)
            # 各ブロックのスパイク数を集計（概算）
            if return_spikes:
                total_spikes = total_spikes + x.sum()

        # [Time, Batch, Seq, Dim] -> [Batch, Seq, Dim]
        # 最終タイムステップの出力を利用
        final_output = x[-1, :, :, :]
        
        logits = self.output_projection(final_output)
        
        # 互換性のための値を返す
        denominator = self.time_steps * batch_size * seq_len
        avg_spikes = total_spikes / denominator if return_spikes and denominator > 0 else torch.tensor(0.0, device=x.device)
        avg_mems = torch.tensor(0.0, device=x.device)

        return logits, avg_spikes, avg_mems

# --- ◾️◾️◾️↑Spiking Transformer アーキテクチャ↑◾️◾️◾️ ---

class SimpleSNN(nn.Module):
    """
    基本的なLIFニューロンで構成されたシンプルなSNNモデル。
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleSNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = LIFNeuron()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = LIFNeuron()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        T, B, _ = x.shape
        self.lif1.reset()
        self.lif2.reset()
        
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
    def __init__(self, config: DictConfig):
        super(SNNCore, self).__init__()
        self.config = config
        model_type = self.config.model.get("type", "simple")

        # mypy: Incompatible types in assignment -> nn.Moduleで汎用化
        self.model: nn.Module

        if model_type == "simple":
            self.model = SimpleSNN(
                input_size=self.config.model.input_size,
                hidden_size=self.config.model.hidden_size,
                output_size=self.config.model.output_size
            )
        elif model_type == "spiking_transformer":
            self.model = SpikingTransformer(**self.config.model.params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x):
        return self.model(x)
