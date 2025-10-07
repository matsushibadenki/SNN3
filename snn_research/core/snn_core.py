# matsushibadenki/snn3/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 改善点:
# 1. 位置エンベディングの追加: 全てのアーキテクチャに位置エンベディングを追加し、シーケンスの順序情報をモデルに供給。
# 2. 学習可能なニューロンパラメータ: AdaptiveLIFNeuronの時定数(tau)と発火閾値(threshold)を学習可能なパラメータに変更。
# 3. 予測符号化の改善: 誤差計算からReLUを削除し、正負両方の誤差を伝播させることで、より豊かな学習信号を生成。
# 4. Spiking Transformerの強化: Transformerブロック内に標準的なFFN(Feed-Forward Network)を追加し、表現力を向上。
# 5. 出力層の安定化: BreakthroughSNNの最終出力層にLayerNormとGELUを追加し、学習を安定化。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional, base
from typing import Tuple, Dict, Any, Optional, List, Type, cast
import math
from omegaconf import DictConfig, OmegaConf


# --- ニューロンモデル ---
class AdaptiveLIFNeuron(base.MemoryModule):
    """
    適応的発火閾値と学習可能な時定数を持つLIFニューロン (表現力向上のための標準ニューロン)
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
        # 膜時定数(tau)と基本閾値(base_threshold)を学習可能なパラメータに変更
        self.tau = nn.Parameter(torch.full((features,), tau))
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.surrogate_function = surrogate.ATan()
        self.register_buffer(
            "adaptive_threshold", torch.full((features,), base_threshold)
        )
        self.adaptive_threshold: torch.Tensor
        self.mem: Optional[torch.Tensor] = None

    def reset(self):
        """ニューロンの状態をリセットする。spikingjellyのreset_netから呼び出される。"""
        super().reset()
        self.mem = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x, device=x.device)

        # 学習可能なtauに基づいて膜電位の減衰率を計算
        mem_decay = torch.exp(-1.0 / self.tau.clamp(min=1.0))
        mem_this_step = self.mem * mem_decay + x
        
        # 学習可能な基本閾値と適応的閾値を合算
        current_threshold = self.base_threshold + self.adaptive_threshold
        spike = self.surrogate_function(mem_this_step - current_threshold)
        
        self.mem = mem_this_step * (1.0 - spike)

        if self.training:
            with torch.no_grad():
                spike_rate_error = spike.mean(dim=0) - self.target_spike_rate
                self.adaptive_threshold += self.adaptation_strength * spike_rate_error
                self.adaptive_threshold.clamp_(min=0.1)

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
            ) for _ in range(self.num_branches)
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
        
        # 改善点: ReLUを削除し、正負両方の誤差を伝播させる
        prediction_error = bottom_up_input - prediction
        prediction_error = self.norm_error(prediction_error)
        
        state_update, inference_mem = self.inference_neuron(self.inference_fc(prediction_error))
        updated_state = self.norm_state(top_down_state + state_update)
        return updated_state, prediction_error, prediction, inference_mem

# --- コアSNNモデル (予測符号化) ---
class BreakthroughSNN(nn.Module):
    """リカレント予測符号化アーキテクチャを実装した階層的SNN"""
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None, max_len: int = 1024):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 改善点: 位置エンベディングを追加
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
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
        # 改善点: 出力層を安定化させるためLayerNormとGELUを追加
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        return_spikes: bool = False,
        return_full_mems: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        
        # 改善点: 位置エンベディングを加算
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb
        
        states = [torch.zeros(batch_size, self.d_state, device=input_ids.device) for _ in range(self.num_layers)]
        
        total_spikes = torch.tensor(0.0, device=input_ids.device)
        total_mem_potential = torch.tensor(0.0, device=input_ids.device)
        all_hidden_states: List[torch.Tensor] = []
        all_mems_list: List[torch.Tensor] = []

        functional.reset_net(self)

        for i in range(seq_len):
            embedded_token = x[:, i, :] # 位置エンベディング込みの入力を利用
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

# --- Spiking Transformer ---
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
        # Softmaxの代わりにSigmoidを使用し、スパイクの性質に合わせる
        attn_weights_spike = torch.sigmoid(attn_scores)

        attn_output = torch.matmul(attn_weights_spike, v_mha)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        
        output = self.out_proj(attn_output)
        output_spike, _ = self.neuron_out(output)
        return output_spike

class STAttenBlock(nn.Module):
    """空間時間アテンションブロック。"""
    def __init__(self, d_model: int, n_head: int, d_ffn: int = 4 * 256):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = SpikeDrivenSelfAttention(d_model, n_head)
        self.norm2 = SNNLayerNorm(d_model)
        
        # 改善点: 標準的なTransformerブロックに倣い、FFN(Feed-Forward Network)を追加
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            AdaptiveLIFNeuron(features=d_ffn),
            nn.Linear(d_ffn, d_model),
            AdaptiveLIFNeuron(features=d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN (LayerNormを先に行う) 形式で安定化
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SpikingTransformer(nn.Module):
    """時間価値を最大化する、空間時間アテンションを備えたSpiking Transformer。"""
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, max_len: int = 1024, **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 改善点: 位置エンベディングを学習可能なパラメータとして追加
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        print(f"🚀 Spiking Transformer (STAtten) initialized with {num_layers} layers.")

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, return_full_mems: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x_embed = self.token_embedding(input_ids)
        
        # 改善点: 位置エンベディングを加算
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = x_embed + pos_emb

        functional.reset_net(self)

        outputs_over_time = []
        spike_outputs_over_time = []

        # 時間ステップにわたってリカレントに処理
        x_t = x
        for t in range(self.time_steps):
            for layer in self.layers:
                x_t = layer(x_t)
            
            spike_outputs_over_time.append(x_t)
            # 時間全体で出力を積分
            if len(outputs_over_time) > 0:
                outputs_over_time.append(outputs_over_time[-1] + x_t)
            else:
                outputs_over_time.append(x_t)
        
        # 最終的な積分値を出力とする
        final_output = outputs_over_time[-1] / self.time_steps

        logits = self.output_projection(final_output)

        total_spikes = torch.tensor(0.0, device=x.device)
        if return_spikes and spike_outputs_over_time:
             total_spikes = torch.stack(spike_outputs_over_time).mean()
        
        avg_spikes = total_spikes
        avg_mems = torch.tensor(0.0, device=x.device)

        return logits, avg_spikes, avg_mems

class SimpleSNN(nn.Module):
    """
    言語モデリングタスク用の、シンプルなリカレントSNN。
    """
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, max_len: int = 1024, **kwargs: Any):
        super(SimpleSNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.lif1 = AdaptiveLIFNeuron(features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        
        x = self.embedding(input_ids)
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb


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
    """
    設定に応じて適切なSNNアーキテクチャをインスタンス化するラッパークラス。
    """
    def __init__(self, config: DictConfig, vocab_size: int):
        super(SNNCore, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        model_type = self.config.get("architecture_type", self.config.get("type", "simple"))

        self.model: nn.Module

        params_untyped = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(params_untyped, Dict):
            raise ValueError(f"Model configuration must be a dictionary. Got: {type(params_untyped)}")
        
        params: Dict[str, Any] = cast(Dict[str, Any], params_untyped)
        # 改善点: max_lenをconfigから取得できるようにし、全モデルに渡す
        max_len = params.get("max_len", 1024)

        if model_type == "predictive_coding":
            self.model = BreakthroughSNN(
                vocab_size=vocab_size,
                d_model=params.get("d_model", 256),
                d_state=params.get("d_state", 128),
                num_layers=params.get("num_layers", 4),
                time_steps=params.get("time_steps", 20),
                n_head=params.get("n_head", 4),
                neuron_config=params.get("neuron", {}),
                max_len=max_len
            )
        elif model_type == "spiking_transformer":
            self.model = SpikingTransformer(
                vocab_size=vocab_size,
                d_model=params.get("d_model", 512),
                n_head=params.get("n_head", 8),
                num_layers=params.get("num_layers", 12),
                time_steps=params.get("time_steps", 32),
                max_len=max_len
            )
        elif model_type == "simple":
            self.model = SimpleSNN(
                vocab_size=vocab_size,
                d_model=params.get("d_model", 128),
                hidden_size=params.get("d_state", 256),
                max_len=max_len
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
