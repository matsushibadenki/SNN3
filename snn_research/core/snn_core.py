# matsushibadenki/snn3/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 重大な問題点の修正:
# 1. 代理勾配関数の修正: 学習信号の弱いSoftSignから、勾配の強いATanに戻す。
# 2. 予測符号化の修正: 誤差計算からReLUを削除し、双方向の誤差を学習できるようにする。
# 3. メモリの勾配切断の修正: .detach()を削除し、BPTTが正しく機能するようにする。
# 4. Spiking Transformerの構造的欠陥の修正: 時間軸の処理を根本的に見直し、トークンごとに時間展開するように修正。
# 5. スパイク統計の正確化: 予測誤差ではなく、実際のスパイクをカウントするように修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional, base  # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type, cast
import math
from omegaconf import DictConfig, OmegaConf


# --- ニューロンモデル ---
class AdaptiveLIFNeuron(base.MemoryModule):
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
        # 1. 代理勾配関数をATanに戻す (alphaで勾配の鋭さを調整)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)
        self.register_buffer(
            "adaptive_threshold", torch.full((features,), base_threshold)
        )
        self.adaptive_threshold: torch.Tensor
        self.mem: Optional[torch.Tensor] = None
        # スパイクを記録するためのバッファ
        self.register_buffer("spikes", torch.zeros(features))

    def reset(self):
        """ニューロンの状態をリセットする。spikingjellyのreset_netから呼び出される。"""
        super().reset()
        self.mem = None
        if hasattr(self, 'spikes'):
            self.spikes.zero_()


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x, device=x.device)
            # spikesバッファもここで初期化
            self.spikes = torch.zeros_like(x, device=x.device)


        mem_this_step = self.mem * self.mem_decay + x
        spike = self.surrogate_function(mem_this_step - self.adaptive_threshold)
        # 5. スパイク統計をバッファに保存
        self.spikes = spike
        
        # 3. .detach() を削除し、BPTTが機能するようにする
        self.mem = mem_this_step * (1.0 - spike)

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
            ) for _ in range(self.num_branches)
        ])
        self.soma_lif = AdaptiveLIFNeuron(branch_features * num_branches)
        self.output_projection = nn.Linear(branch_features * num_branches, input_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_outputs = [branch(x)[0] for branch in self.branches]
        concatenated_spikes = torch.cat(branch_outputs, dim=-1)
        soma_spike, soma_mem = self.soma_lif(concatenated_spikes)
        # 出力はスパイクではなく、次の層への入力となる電位として扱う
        output = self.output_projection(soma_mem)
        return output, soma_mem

class SNNLayerNorm(nn.Module):
    """SNNのためのタイムステップごとに行うLayerNorm"""
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
    def forward(self, x):
        # スパイクではなく、膜電位や入力電流のような連続値に適用する
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
        # 膜電位に正規化を適用
        prediction, _ = self.generative_neuron(self.generative_fc(self.norm_state(top_down_state)))
        # 2. 予測誤差の計算でReLUを削除
        prediction_error = bottom_up_input - prediction
        
        state_update, inference_mem = self.inference_neuron(self.inference_fc(self.norm_error(prediction_error)))
        updated_state = top_down_state + state_update
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
        self.output_projection = nn.Linear(d_state, vocab_size)

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
            
            # 各トークンに対して時間ステップを展開
            for t in range(self.time_steps):
                for j in range(self.num_layers):
                    states[j], error, _, mem = self.pc_layers[j](bottom_up_input, states[j])
                    bottom_up_input = error
                    
                    if return_spikes:
                        # inference_neuronがAdaptiveLIFNeuronのインスタンスであると仮定
                         total_spikes_val += self.pc_layers[j].inference_neuron.spikes.sum()

            # 最上位層の状態をそのトークンの表現として保存
            all_hidden_states.append(states[-1])


        final_hidden_states = torch.stack(all_hidden_states, dim=1)
        
        if output_hidden_states:
            final_output = final_hidden_states
        else:
            final_output = self.output_projection(final_hidden_states)

        avg_spikes = total_spikes_val / (seq_len * self.time_steps * batch_size) if return_spikes and seq_len > 0 else torch.tensor(0.0)
        final_mem = torch.tensor(0.0) # Dummy

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
    """空間時間アテンションブロック。"""
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

class SpikingTransformer(nn.Module):
    """時間価値を最大化する、空間時間アテンションを備えたSpiking Transformer。"""
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        print(f"🚀 Spiking Transformer (STAtten) initialized with {num_layers} layers.")

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, return_full_mems: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x_embed = self.token_embedding(input_ids)
        x_embed = x_embed + self.pos_embedding[:, :seq_len, :]

        functional.reset_net(self)

        token_outputs = []
        total_spikes_val = 0

        # 4. Spiking Transformerの時間軸の誤りを修正
        # トークンシーケンスをループ
        for i in range(seq_len):
            # 現在のトークンの埋め込み
            x_token = x_embed[:, i, :]
            
            # 時間ステップで展開
            for t in range(self.time_steps):
                for layer in self.layers:
                    x_token = layer(x_token)
            
            token_outputs.append(x_token)
            if return_spikes:
                for layer in self.layers:
                    # 各LIFニューロンからスパイク数を集計
                    total_spikes_val += layer.lif1.spikes.sum()
                    total_spikes_val += layer.lif2.spikes.sum()
                    total_spikes_val += layer.lif3.spikes.sum()


        final_output = torch.stack(token_outputs, dim=1)
        logits = self.output_projection(final_output)

        avg_spikes = total_spikes_val / (seq_len * self.time_steps * batch_size) if return_spikes and seq_len > 0 else torch.tensor(0.0)
        avg_mems = torch.tensor(0.0, device=x_embed.device)

        return logits, avg_spikes, avg_mems

class SimpleSNN(nn.Module):
    """
    言語モデリングタスク用の、シンプルなリカレントSNN。
    """
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, **kwargs: Any):
        super(SimpleSNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.lif1 = AdaptiveLIFNeuron(features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

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
                vocab_size=vocab_size,
                d_model=params.get("d_model", 128),
                hidden_size=params.get("d_state", 256)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
