# matsushibadenki/snn3/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 専門家の指摘に基づく抜本的修正:
# 1. BugFix: 各モデルクラスの__init__に**kwargsを追加し、不要な引数を無視するように。
# 2. 予測符号化の修正: ReLUを削除し、学習可能な誤差スケールを追加。
# 3. Spiking Transformerの構造的欠陥の修正: 時間軸の処理を根本的に見直す。
# 4. 適切な重み初期化を追加。
# 5. LayerNormの適用箇所を修正。
# 6. スパイク統計の収集方法を正確化。
# 7. ニューロン実装を `neurons.py` からインポート。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type, cast
import math
from omegaconf import DictConfig, OmegaConf

# 外部ファイルからニューロンをインポート
from .neurons import AdaptiveLIFNeuron

# --- レイヤーとモジュール ---

class SNNLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # スパイクではなく、膜電位や入力電流のような連続値に適用
        return self.norm(x)

class PredictiveCodingLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_neuron = neuron_class(features=d_model, **neuron_params)

        self.inference_fc = nn.Linear(d_model, d_state)
        self.inference_neuron = neuron_class(features=d_state, **neuron_params)
        
        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)
        self.error_scale = nn.Parameter(torch.ones(1))

    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 予測生成: 状態を正規化してから実行
        prediction, _ = self.generative_neuron(self.generative_fc(self.norm_state(top_down_state)))
        # 誤差計算: ReLUを削除し、学習可能なスケールを適用
        prediction_error = (bottom_up_input - prediction) * self.error_scale
        
        # 状態更新: 誤差を正規化してから実行
        state_update, _ = self.inference_neuron(self.inference_fc(self.norm_error(prediction_error)))
        updated_state = top_down_state + state_update
        return updated_state, prediction_error, prediction

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
        # (B, N, C) -> (B, N, C)
        q, _ = self.neuron_q(self.q_proj(x))
        k, _ = self.neuron_k(self.k_proj(x))
        v = self.v_proj(x)

        q = q.view(B, N, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, N, self.n_head, self.d_head).permute(0, 2, 3, 1)
        v = v.view(B, N, self.n_head, self.d_head).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k) / math.sqrt(self.d_head)
        attn_weights = torch.sigmoid(attn_scores)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        
        return self.out_proj(attn_output)

# 修正5: STAttenBlockのバッチ処理
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
        """
        Process input with self-attention and feedforward layers.
        
        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Output of shape (batch_size, seq_len, d_model).
        """
        B, T, D = x.shape
        
        # Self-attention branch (operates on full sequence)
        attn_out = self.attn(self.norm1(x))  # (B, T, D)
        x_attn = x + attn_out  # (B, T, D)
        
        # Apply LIF neuron token-by-token
        x_attn_spikes = []
        for t_idx in range(T):
            spike, _ = self.lif1(x_attn[:, t_idx, :])  # (B, D)
            x_attn_spikes.append(spike)
        x_res = torch.stack(x_attn_spikes, dim=1)  # (B, T, D)
        
        # Feedforward branch
        ffn_in = self.norm2(x_res)  # (B, T, D)
        
        # Process token-by-token through FFN
        ffn_outputs = []
        for t_idx in range(T):
            ffn_hidden, _ = self.lif2(self.fc1(ffn_in[:, t_idx, :]))  # (B, 4D)
            ffn_out = self.fc2(ffn_hidden)  # (B, D)
            ffn_outputs.append(ffn_out)
        ffn_out = torch.stack(ffn_outputs, dim=1)  # (B, T, D)
        
        # Final residual connection
        x_ffn = x_res + ffn_out  # (B, T, D)
        out_spikes = []
        for t_idx in range(T):
            spike, _ = self.lif3(x_ffn[:, t_idx, :])  # (B, D)
            out_spikes.append(spike)
        out = torch.stack(out_spikes, dim=1)  # (B, T, D)
        
        return out

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

    def get_total_spikes(self):
        """モデル全体の総スパイク数を収集する。"""
        total_spikes = 0
        for module in self.modules():
            if isinstance(module, AdaptiveLIFNeuron):
                total_spikes += module.spikes.sum()
        return total_spikes

# --- モデル定義 ---
class BreakthroughSNN(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.input_encoder = nn.Linear(d_model, d_model)
        
        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, AdaptiveLIFNeuron, neuron_config if neuron_config is not None else {}) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_state * num_layers, vocab_size)
        self._init_weights()

    # 修正2: BreakthroughSNNの時間軸処理
    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with corrected temporal processing.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            return_spikes (bool): Whether to return spike statistics.
            **kwargs: Additional arguments (ignored for compatibility).
        
        Returns:
            Tuple containing:
                - logits (torch.Tensor): Output logits of shape (batch_size, seq_len, vocab_size).
                - avg_spikes (torch.Tensor): Average spike rate.
                - placeholder (torch.Tensor): Placeholder for compatibility.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed all tokens at once
        token_emb = self.token_embedding(input_ids)  # (B, T, D)
        embedded_sequence = self.input_encoder(token_emb)  # (B, T, D)
        
        # Initialize states once for the entire sequence
        inference_neuron = cast(AdaptiveLIFNeuron, self.pc_layers[0].inference_neuron)
        states = [torch.zeros(batch_size, inference_neuron.features, device=device) 
                  for _ in range(self.num_layers)]
        
        # Process the entire sequence through time
        all_timestep_outputs = []
        
        for t in range(self.time_steps):
            sequence_outputs = []
            
            for i in range(seq_len):
                bottom_up_input = embedded_sequence[:, i, :]
                
                # Process through all layers
                for j in range(self.num_layers):
                    states[j], error, _ = self.pc_layers[j](bottom_up_input, states[j])
                    bottom_up_input = error
                
                # Collect layer states
                sequence_outputs.append(torch.cat(states, dim=1))
            
            all_timestep_outputs.append(torch.stack(sequence_outputs, dim=1))
        
        # Use final timestep output (or average across timesteps)
        final_hidden_states = all_timestep_outputs[-1]  # (B, T, D_state*num_layers)
        
        # Project to vocabulary
        logits = self.output_projection(final_hidden_states)  # (B, T, vocab_size)
        
        # Calculate spike statistics
        total_spikes = self.get_total_spikes()
        avg_spikes = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else torch.tensor(0.0, device=device)
        
        return logits, avg_spikes, torch.tensor(0.0, device=device)

class SpikingTransformer(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head) for _ in range(num_layers)])
        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        print(f"🚀 Spiking Transformer (STAtten) initialized with {num_layers} layers.")
        self._init_weights()

    # 修正3: SpikingTransformerの構造改善
    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with corrected self-attention over the entire sequence.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            return_spikes (bool): Whether to return spike statistics.
            **kwargs: Additional arguments (ignored for compatibility).
        
        Returns:
            Tuple containing:
                - logits (torch.Tensor): Output logits of shape (batch_size, seq_len, vocab_size).
                - avg_spikes (torch.Tensor): Average spike rate.
                - placeholder (torch.Tensor): Placeholder for compatibility.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed tokens with positional encoding
        x = self.token_embedding(input_ids)  # (B, T, D)
        x = x + self.pos_embedding[:, :seq_len, :]  # (B, T, D)
        
        # Process the entire sequence through time
        for t in range(self.time_steps):
            for layer in self.layers:
                x = layer(x)  # (B, T, D) - 自己注意機構がシーケンス全体を考慮
        
        # Project to vocabulary
        x_normalized = self.final_norm(x)  # (B, T, D)
        logits = self.output_projection(x_normalized)  # (B, T, vocab_size)
        
        # Calculate spike statistics
        total_spikes = self.get_total_spikes()
        avg_spikes = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else torch.tensor(0.0, device=device)
        
        return logits, avg_spikes, torch.tensor(0.0, device=device)

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
        
        outputs = []
        functional.reset_net(self)
        for t in range(T):
            x_t = x[:, t, :]
            out, _ = self.lif1(self.fc1(x_t))
            out = self.fc2(out)
            outputs.append(out)
            
        logits = torch.stack(outputs, dim=1)
        avg_spikes = self.get_total_spikes() / (B * T) if return_spikes else torch.tensor(0.0)
        return logits, avg_spikes, torch.tensor(0.0)

class SNNCore(nn.Module):
    def __init__(self, config: DictConfig, vocab_size: int):
        super(SNNCore, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        model_type = self.config.get("architecture_type", "simple")
        self.model: nn.Module
        params: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(self.config, resolve=True))
        params.pop('path', None)
        neuron_config = params.pop('neuron', {})

        model_map = {
            "predictive_coding": BreakthroughSNN,
            "spiking_transformer": SpikingTransformer,
            "simple": SimpleSNN
        }
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = model_map[model_type](vocab_size=vocab_size, neuron_config=neuron_config, **params)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
