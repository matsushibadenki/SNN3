# matsushibadenki/snn3/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 専門家の指摘に基づく抜本的修正:
# 1. BugFix: 各モデルクラスの__init__に**kwargsを追加し、不要な引数を無視するように。
# 2. 代理勾配関数の修正: 学習信号の強いATanに戻す。
# 3. 予測符号化の修正: ReLUを削除し、学習可能な誤差スケールを追加。
# 4. メモリの勾配切断の修正: .detach()を削除し、BPTTを正しく機能させる。
# 5. Spiking Transformerの構造的欠陥の修正: 時間軸の処理を根本的に見直す。
# 6. spikesバッファのshapeを統一。
# 7. 適切な重み初期化を追加。
# 8. LayerNormの適用箇所を修正。
# 9. スパイク統計の収集方法を正確化。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional, base
from typing import Tuple, Dict, Any, Optional, List, Type, cast
import math
from omegaconf import DictConfig, OmegaConf


# --- ニューロンモデル ---
class AdaptiveLIFNeuron(base.MemoryModule):
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
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)
        
        self.register_buffer("adaptive_threshold", torch.full((features,), base_threshold))
        self.mem: Optional[torch.Tensor] = None
        self.register_buffer("spikes", torch.zeros(features))

    def reset(self):
        super().reset()
        self.mem = None
        if hasattr(self, 'spikes'):
            self.spikes.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x, device=x.device)

        mem_this_step = self.mem * self.mem_decay + x
        spike = self.surrogate_function(mem_this_step - self.adaptive_threshold)
        
        # spikesバッファのshapeを統一
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike

        # .detach()を削除してBPTTを有効化
        self.mem = mem_this_step * (1.0 - spike)

        # 適応閾値の更新は勾配計算外で行う
        with torch.no_grad():
            if self.training:
                spike_rate_error = spike.mean() - self.target_spike_rate
                self.adaptive_threshold += self.adaptation_strength * spike_rate_error
                self.adaptive_threshold.clamp_(min=0.5)

        return spike, mem_this_step

class DendriticNeuron(nn.Module):
    def __init__(self, input_features: int, num_branches: int, branch_features: int):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_features, branch_features),
                AdaptiveLIFNeuron(branch_features)
            ) for _ in range(self.num_branches)
        ])
        self.soma_lif = AdaptiveLIFNeuron(branch_features * num_branches)
        self.output_projection = nn.Linear(branch_features * num_branches, input_features)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_outputs = [branch(x)[0] for branch in self.branches]
        concatenated_spikes = torch.cat(branch_outputs, dim=-1)
        soma_spike, soma_mem = self.soma_lif(concatenated_spikes)
        output = self.output_projection(soma_mem)
        return output, soma_mem

class SNNLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

# --- 予測符号化レイヤー ---
class PredictiveCodingLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_neuron = neuron_class(features=d_model, **neuron_params) if neuron_class == AdaptiveLIFNeuron else neuron_class(input_features=d_model, **neuron_params)

        self.inference_fc = nn.Linear(d_model, d_state)
        self.inference_neuron = neuron_class(features=d_state, **neuron_params) if neuron_class == AdaptiveLIFNeuron else neuron_class(input_features=d_state, **neuron_params)
        
        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)
        self.error_scale = nn.Parameter(torch.ones(1))


    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, _ = self.generative_neuron(self.generative_fc(self.norm_state(top_down_state)))
        prediction_error = (bottom_up_input - prediction) * self.error_scale
        
        state_update, inference_mem = self.inference_neuron(self.inference_fc(self.norm_error(prediction_error)))
        updated_state = top_down_state + state_update
        return updated_state, prediction_error, prediction, inference_mem

# --- モデルの基底クラス ---
class BaseModel(nn.Module):
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)


# --- コアSNNモデル (予測符号化) ---
class BreakthroughSNN(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        neuron_config = neuron_config or {"type": "lif"}
        neuron_class = AdaptiveLIFNeuron
        neuron_params = {}
        self.input_encoder = nn.Sequential(nn.Linear(d_model, d_model), AdaptiveLIFNeuron(features=d_model))
        
        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, n_head, neuron_class, neuron_params) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_state, vocab_size)
        self._init_weights()

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
        
        total_spikes_val = torch.tensor(0.0, device=input_ids.device)
        all_hidden_states: List[torch.Tensor] = []

        functional.reset_net(self)

        for i in range(seq_len):
            embedded_token = token_emb[:, i, :]
            bottom_up_input, _ = self.input_encoder(embedded_token)
            
            for t in range(self.time_steps):
                for j in range(self.num_layers):
                    states[j], error, _, mem = self.pc_layers[j](bottom_up_input, states[j])
                    bottom_up_input = error
                    
                    if return_spikes:
                         total_spikes_val += self.pc_layers[j].inference_neuron.spikes.sum()

            all_hidden_states.append(states[-1])

        final_hidden_states = torch.stack(all_hidden_states, dim=1)
        
        if output_hidden_states:
            final_output = final_hidden_states
        else:
            final_output = self.output_projection(final_hidden_states)

        avg_spikes = total_spikes_val / (seq_len * self.time_steps * batch_size) if return_spikes and seq_len > 0 else torch.tensor(0.0)
        final_mem = torch.tensor(0.0)

        return final_output, avg_spikes, final_mem

# --- Spiking Transformer ---
class SpikeDrivenSelfAttention(nn.Module):
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
        return output
        
class STAttenBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = SpikeDrivenSelfAttention(d_model, n_head)
        self.lif1 = AdaptiveLIFNeuron(features=d_model)
        self.norm2 = SNNLayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.lif2 = AdaptiveLIFNeuron(features=d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.lif3 = AdaptiveLIFNeuron(features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x_plus_attn, _ = self.lif1(x + attn_out)

        ffn_out = self.fc2(self.lif2(self.fc1(self.norm2(x_plus_attn)))[0])
        out, _ = self.lif3(x_plus_attn + ffn_out)
        return out

class SpikingTransformer(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        print(f"🚀 Spiking Transformer (STAtten) initialized with {num_layers} layers.")
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, return_full_mems: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x_embed = self.token_embedding(input_ids)
        x_embed = x_embed + self.pos_embedding[:, :seq_len, :]

        functional.reset_net(self)

        token_outputs = []
        total_spikes_val = 0

        for i in range(seq_len):
            x_token = x_embed[:, i, :]
            
            for t in range(self.time_steps):
                for layer in self.layers:
                    x_token = layer(x_token)
            
            token_outputs.append(x_token)
            if return_spikes:
                for layer in self.layers:
                    total_spikes_val += layer.lif1.spikes.sum() + layer.lif2.spikes.sum() + layer.lif3.spikes.sum()


        final_output = torch.stack(token_outputs, dim=1)
        logits = self.output_projection(final_output)

        avg_spikes = total_spikes_val / (seq_len * self.time_steps * batch_size) if return_spikes and seq_len > 0 else torch.tensor(0.0)
        avg_mems = torch.tensor(0.0, device=x_embed.device)

        return logits, avg_spikes, avg_mems

class SimpleSNN(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, **kwargs: Any):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.lif1 = AdaptiveLIFNeuron(features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        
        x = self.embedding(input_ids)
        functional.reset_net(self)
        
        outputs = []
        total_spikes = torch.tensor(0.0, device=input_ids.device)
        
        for t in range(T):
            x_t = x[:, t, :]
            out = self.fc1(x_t)
            spikes, _ = self.lif1(out)
            out = self.fc2(spikes)
            
            outputs.append(out)
            if return_spikes:
                total_spikes += spikes.sum()

        logits = torch.stack(outputs, dim=1)
        
        avg_spikes = total_spikes / (B * T) if return_spikes and B * T > 0 else torch.tensor(0.0)
        mem = torch.tensor(0.0, device=input_ids.device) 

        return logits, avg_spikes, mem

class SNNCore(nn.Module):
    def __init__(self, config: DictConfig, vocab_size: int):
        super(SNNCore, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        model_type = self.config.get("architecture_type", self.config.get("type", "simple"))
        self.model: nn.Module
        params: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(self.config, resolve=True))

        params.pop('path', None)

        if model_type == "predictive_coding":
            self.model = BreakthroughSNN(vocab_size=vocab_size, **params)
        elif model_type == "spiking_transformer":
            self.model = SpikingTransformer(vocab_size=vocab_size, **params)
        elif model_type == "simple":
            self.model = SimpleSNN(vocab_size=vocab_size, **params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
