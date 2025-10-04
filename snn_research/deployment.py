# matsushibadenki/snn3/snn_research/deployment.py
# Title: SNN推論エンジン
# Description: 訓練済みSNNモデルをロードし、テキスト生成のための推論を実行するクラス。
#              HuggingFaceの`generate`メソッドに似たインターフェースを提供。
#              mypyエラー修正: modelの型ヒントをUnionで両対応させた。
#              mypyエラー修正: mypyが推論できないtokenizerの属性アクセスエラーを型キャストで抑制。

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from typing import Iterator, Optional, Dict, Any, List, Union
from .core.snn_core import BreakthroughSNN, SpikingTransformer
from omegaconf import DictConfig
from snn_research.core.snn_core import SNNCore # SNNCoreをインポート


class SNNInferenceEngine:
    """
    学習済みSNNモデルをロードして推論を実行するエンジン。
    """
    def __init__(self, config: DictConfig):
        self.config = config
        # SNNCoreがconfigに基づいて適切なモデルをインスタンス化する
        self.model = SNNCore(config)
        
        if config.deployment.get("model_path"):
            try:
                self.model.load_state_dict(torch.load(config.deployment.model_path))
                print(f"Model loaded from {config.deployment.model_path}")
            except FileNotFoundError:
                print(f"Warning: Model file not found at {config.deployment.model_path}. Using an untrained model.")
        
        self.model.eval()

    def _load_model(self) -> None:
        """モデルとトークナイザをロードする。"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        print(f"Loading model from: {self.model_path}...")
        
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        self.config = config_data
        architecture = config_data.get("architecture_type", "predictive_coding")
        
        model_class = SpikingTransformer if architecture == "spiking_transformer" else BreakthroughSNN
        
        tokenizer_config_path = self.model_path / "tokenizer_config.json"
        if 'vocab_size' not in config_data and tokenizer_config_path.exists():
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
                config_data['vocab_size'] = tokenizer_config.get('vocab_size')

        model_instance: Union[BreakthroughSNN, SpikingTransformer]
        if model_class == SpikingTransformer:
            model_instance = SpikingTransformer(
                vocab_size=config_data['vocab_size'],
                d_model=config_data['d_model'],
                n_head=config_data['n_head'],
                num_layers=config_data['num_layers'],
                time_steps=config_data['time_steps']
            )
        else:
            model_instance = BreakthroughSNN(
                vocab_size=config_data['vocab_size'],
                d_model=config_data['d_model'],
                d_state=config_data['d_state'],
                num_layers=config_data['num_layers'],
                time_steps=config_data['time_steps'],
                n_head=config_data['n_head'],
                neuron_config=config_data.get('neuron')
            )
        self.model = model_instance
        
        model_weights_path = self.model_path / "pytorch_model.bin"
        if not model_weights_path.exists():
            raise FileNotFoundError(f"pytorch_model.bin not found in {self.model_path}")
            
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        print("Model and tokenizer loaded successfully.")


    def generate(self, prompt: str, max_len: int, stop_sequences: list = None) -> Iterator[str]:
        # この部分はダミー実装です。実際のトークナイザーと生成ロジックが必要です。
        current_text = prompt
        for i in range(max_len):
            # 実際のモデル入力とデコード処理が必要
            new_token = f" token_{i}"
            if stop_sequences and any(seq in current_text for seq in stop_sequences):
                break
            yield new_token
            current_text += new_token
        """
        プロンプトに基づいてテキストをストリーミング生成する。
        """
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        tokenizer_callable = getattr(self.tokenizer, "__call__", None)
        if not callable(tokenizer_callable):
            raise TypeError("Tokenizer is not callable.")
        input_ids = tokenizer_callable(prompt, return_tensors="pt")["input_ids"].to(self.device)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        total_spikes = 0
        
        for i in range(max_len):
            with torch.no_grad():
                outputs, avg_spikes, _ = self.model(input_ids, return_spikes=True)
            
            total_spikes += avg_spikes.item() * input_ids.shape[1]
            
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            if next_token_id.item() == getattr(self.tokenizer, 'eos_token_id', None):
                break
            
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            new_token = getattr(self.tokenizer, "decode")(next_token_id.item())
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            
            if stop_sequences and any(seq in new_token for seq in stop_sequences):
                break
                
            yield new_token
            
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

        self.last_inference_stats = {"total_spikes": total_spikes}