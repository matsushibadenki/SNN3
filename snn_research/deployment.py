# matsushibadenki/snn3/snn_research/deployment.py
# Title: SNN推論エンジン
# Description: 訓練済みSNNモデルをロードし、テキスト生成のための推論を実行するクラス。
#              HuggingFaceの`generate`メソッドに似たインターフェースを提供。
#              mypyエラー修正: modelの型ヒントをUnionで両対応させた。

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from typing import Iterator, Optional, Dict, Any, List, Union

from .core.snn_core import BreakthroughSNN, SpikingTransformer

class SNNInferenceEngine:
    """訓練済みSNNモデルの推論を実行するエンジン。"""
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = Path(model_path)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.model: Union[BreakthroughSNN, SpikingTransformer]
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.tokenizer: AutoTokenizer
        self.config: Dict[str, Any]
        self._load_model()
        
        self.last_inference_stats: Dict[str, Any] = {}

    def _load_model(self) -> None:
        """モデルとトークナイザをロードする。"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        print(f"Loading model from: {self.model_path}...")
        
        # config.jsonからアーキテクチャを特定
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        self.config = config_data
        architecture = config_data.get("architecture_type", "predictive_coding") # デフォルトはBreakthroughSNN
        
        model_class = SpikingTransformer if architecture == "spiking_transformer" else BreakthroughSNN
        
        # from_pretrainedを使用してモデルをロード
        # この部分はカスタムPreTrainedModelクラスの `from_pretrained` の実装に依存
        # ここでは簡略化のため、直接インスタンス化してstate_dictをロードする
        
        # 1. モデルのインスタンス化
        # vocab_sizeなどの必須パラメータをconfigから取得
        # vocab_sizeがconfigになければ、tokenizer_config.jsonから取得
        tokenizer_config_path = self.model_path / "tokenizer_config.json"
        if 'vocab_size' not in config_data and tokenizer_config_path.exists():
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
                config_data['vocab_size'] = tokenizer_config.get('vocab_size')

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # モデルクラスに応じて初期化
        if model_class == SpikingTransformer:
            # SpikingTransformerが必要とする引数をconfig_dataから渡す
            model_instance = SpikingTransformer(
                vocab_size=config_data['vocab_size'],
                d_model=config_data['d_model'],
                n_head=config_data['n_head'],
                num_layers=config_data['num_layers'],
                time_steps=config_data['time_steps']
            )
        else: # BreakthroughSNN
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
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        # 2. state_dictのロード
        model_weights_path = self.model_path / "pytorch_model.bin"
        if not model_weights_path.exists():
            raise FileNotFoundError(f"pytorch_model.bin not found in {self.model_path}")
            
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 3. トークナイザのロード
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        print("Model and tokenizer loaded successfully.")


    def generate(
        self,
        prompt: str,
        max_len: int,
        stop_sequences: Optional[List[str]] = None
    ) -> Iterator[str]:
        """
        プロンプトに基づいてテキストをストリーミング生成する。
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        total_spikes = 0
        
        for i in range(max_len):
            with torch.no_grad():
                outputs, avg_spikes, _ = self.model(input_ids, return_spikes=True)
            
            total_spikes += avg_spikes.item() * input_ids.shape[1]
            
            # 最後のトークンのロジットから次のトークンを予測
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # EOSトークンが出たら終了
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            
            # 生成されたトークンをデコード
            new_token = self.tokenizer.decode(next_token_id.item())
            
            # ストップシーケンスのチェック
            if stop_sequences and any(seq in new_token for seq in stop_sequences):
                break
                
            yield new_token
            
            # 入力を更新
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

        self.last_inference_stats = {"total_spikes": total_spikes}
