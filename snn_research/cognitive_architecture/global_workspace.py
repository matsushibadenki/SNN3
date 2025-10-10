# matsushibadenki/snn3/snn_research/cognitive_architecture/global_workspace.py
# Title: グローバルワークスペース
# Description: 異なる認知モジュール間で情報を共有・統合するための中央ハブ。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
# 改善点:
# - ROADMAPフェーズ4「スパイクベース通信プロトコル」に基づき、SpikeEncoderDecoderを導入。
# - ワークスペース内の全ての情報交換を、抽象データではなくスパイクパターンで行うように変更。

from typing import Dict, Any, List, Callable
import random # randomをインポート

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.communication.spike_encoder_decoder import SpikeEncoderDecoder

class GlobalWorkspace:
    """
    認知アーキテクチャ全体で情報をスパイクパターンとして共有するための中央情報ハブ。
    """
    def __init__(self, model_registry: ModelRegistry):
        self.blackboard: Dict[str, Any] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.model_registry = model_registry
        self.encoder_decoder = SpikeEncoderDecoder()

    def broadcast(self, source: str, data: Any) -> None:
        """
        情報をスパイクパターンにエンコードしてブラックボードに書き込み、購読者に通知する。
        """
        print(f"[GlobalWorkspace] Encoding and broadcasting from '{source}': {str(data)[:100]}...")
        # データをスパイクパターンにエンコード
        if isinstance(data, dict):
            spiked_data = self.encoder_decoder.encode_dict_to_spikes(data)
        elif isinstance(data, str):
            spiked_data = self.encoder_decoder.encode_text_to_spikes(data)
        else:
            # その他のデータ型は文字列に変換してエンコード
            spiked_data = self.encoder_decoder.encode_text_to_spikes(str(data))
            
        self.blackboard[source] = spiked_data
        self._notify(source)

    def subscribe(self, source: str, callback: Callable) -> None:
        """
        特定のソースからの情報更新を購読する。
        """
        if source not in self.subscribers:
            self.subscribers[source] = []
        self.subscribers[source].append(callback)

    def _notify(self, source: str) -> None:
        """
        更新があったソースの購読者に、デコードした情報を通知する。
        """
        if source in self.subscribers:
            decoded_info = self.get_information(source)
            for callback in self.subscribers[source]:
                callback(decoded_info)

    def get_information(self, source: str) -> Any:
        """
        ブラックボードからスパイクパターンを取得し、デコードして返す。
        """
        spiked_data = self.blackboard.get(source)
        if spiked_data is None:
            return None
        
        # まず辞書としてデコードを試みる
        decoded_data = self.encoder_decoder.decode_spikes_to_dict(spiked_data)
        if isinstance(decoded_data, dict) and "error" in decoded_data:
            # 辞書へのデコードが失敗した場合、単純なテキストとしてデコードする
            return self.encoder_decoder.decode_spikes_to_text(spiked_data)
        return decoded_data


    def get_full_context(self) -> Dict[str, Any]:
        """
        現在のワークスペースの全コンテキストをデコードして取得する。
        """
        decoded_context = {}
        for source, spiked_data in self.blackboard.items():
            decoded_context[source] = self.get_information(source)
        return decoded_context
