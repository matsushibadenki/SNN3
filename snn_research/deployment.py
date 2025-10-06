# matsushibadenki/snn3/snn_research/deployment.py
# Title: SNN推論エンジン
# Description: 訓練済みSNNモデルをロードし、テキスト生成のための推論を実行するクラス。
#              HuggingFaceの`generate`メソッドに似たインターフェースを提供。
#              mypyエラー修正: modelの型ヒントをUnionで両対応させた。
#              mypyエラー修正: mypyが推論できないtokenizerの属性アクセスエラーを型キャストで抑制。
#              mypyエラー修正: 未定義属性(model_path, device)を修正し、_load_modelを統合。

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from typing import Iterator, Optional, Dict, Any, List, Union
from .core.snn_core import BreakthroughSNN, SpikingTransformer
from omegaconf import DictConfig, OmegaConf
from snn_research.core.snn_core import SNNCore # SNNCoreをインポート


class SNNInferenceEngine:
    """
    学習済みSNNモデルをロードして推論を実行するエンジン。
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = config.deployment.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 先にTokenizerをロードしてvocab_sizeを取得
        tokenizer_path = config.deployment.get("tokenizer_path", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Could not load tokenizer from {tokenizer_path}. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # vocab_sizeを渡してSNNCoreを初期化
        vocab_size = len(self.tokenizer)
        self.model = SNNCore(config, vocab_size=vocab_size)
        
        model_path = config.deployment.get("model_path")
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                # 'model.' プレフィックスが付いている場合、それを取り除く
                if all(key.startswith('model.') for key in state_dict.keys()):
                    state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}
                
                # SNNCoreがラップしている内部モデルのstate_dictをロードする
                self.model.model.load_state_dict(state_dict)
                print(f"Model loaded from {model_path}")
            except FileNotFoundError:
                print(f"Warning: Model file not found at {model_path}. Using an untrained model.")
            except RuntimeError as e:
                print(f"Warning: Failed to load state_dict, possibly due to architecture mismatch: {e}. Using an untrained model.")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_len: int, stop_sequences: Optional[List[str]] = None) -> Iterator[str]:
        """
        プロンプトに基づいてテキストをストリーミング生成する。
        """
        tokenizer_callable = getattr(self.tokenizer, "__call__", None)
        if not callable(tokenizer_callable):
            raise TypeError("Tokenizer is not callable.")
        input_ids = tokenizer_callable(prompt, return_tensors="pt")["input_ids"].to(self.device)
        
        total_spikes = 0
        
        for i in range(max_len):
            with torch.no_grad():
                outputs, avg_spikes, _ = self.model(input_ids, return_spikes=True)
            
            total_spikes += avg_spikes.item() * input_ids.shape[1]
            
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            if next_token_id.item() == getattr(self.tokenizer, 'eos_token_id', None):
                break
            
            new_token = getattr(self.tokenizer, "decode")(next_token_id.item())
            
            if stop_sequences and any(seq in new_token for seq in stop_sequences):
                break
                
            yield new_token
            
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

        self.last_inference_stats = {"total_spikes": total_spikes}
