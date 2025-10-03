# matsushibadenki/snn3/benchmark/ann_baseline.py
#
# SNNモデルとの性能比較を行うためのANNベースラインモデル
#
# 目的:
# - ロードマップ フェーズ1「1.2. ANNベースラインとの比較」に対応。
# - SNNとほぼ同等のパラメータ数を持つ標準的なANNモデルを実装し、
#   公平な性能比較の土台を築く。
#
# アーキテクチャ:
# - 事前学習済みモデルは使用せず、スクラッチで学習するシンプルなTransformerエンコーダを採用。
# - 単語埋め込み層 + Transformerエンコーダ層 + 分類ヘッドという標準的な構成。

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Tuple

class ANNBaselineModel(nn.Module):
    """
    シンプルなTransformerベースのテキスト分類モデル。
    BreakthroughSNNとの比較用。
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int, nlayers: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformerエンコーダ層を定義
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        # 分類ヘッド
        self.classifier = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids (torch.Tensor): 入力シーケンス (batch_size, seq_len)
            attention_mask (torch.Tensor): パディングマスク (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 分類ロジットとNone (SNNとの互換性のため)
        """
        # 互換性のための引数名マッピング
        src = input_ids
        # 注意: TransformerEncoder は (N, E) または (S, N, E) を期待します。
        # attention_mask は src_key_padding_mask として使用されます。
        # (batch_size, seq_len) の形状で、パディング位置がTrueである必要があります。
        # 現在のTokenizerはパディング位置を1、非パディングを0としていますが、
        # PyTorchのTransformerは逆（パディングがTrue）を期待するため、変換が必要です。
        src_key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Transformerエンコーダに入力
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        # パディングを考慮した平均プーリング
        if attention_mask is not None:
            # attention_maskはパディングが0なので、それを反転してマスクとして使用
            mask = attention_mask.unsqueeze(-1).expand_as(encoded)
            masked_encoded = encoded * mask.float()
            pooled = masked_encoded.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1e-9)
        else:
            pooled = encoded.mean(dim=1)

        logits = self.classifier(pooled)
        # SNN評価との互換性のため、タプルで返す
        return logits, None

