# snn_research/cognitive_architecture/symbol_grounding.py
# (新規作成)
#
# Title: 記号創発システム (Symbol Grounding System)
#
# Description:
# - ROADMAPフェーズ7「記号的飛躍」で提案された「記号創発」機能を実装。
# - 未知の観測（stableなニューロン発火パターンなど）に対して、
#   新しいシンボル（例: "concept_101"）を自律的に割り当てる。
# - 生成したシンボルと観測内容を関連付け、RAGSystemのナレッジグラフに記録する。

from typing import Set, Dict, Any
import hashlib
from .rag_snn import RAGSystem

class SymbolGrounding:
    """
    観測から新しいシンボルを創発し、ナレッジグラフに定着させるシステム。
    """
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.known_concepts: Set[str] = set()
        self.concept_counter = 100 # concept_100から開始

    def _get_observation_hash(self, observation: Dict[str, Any]) -> str:
        """観測内容から一意のハッシュを生成する"""
        # 辞書を安定した文字列に変換してハッシュ化
        s = str(sorted(observation.items()))
        return hashlib.sha256(s.encode()).hexdigest()

    def process_observation(self, observation: Dict[str, Any], context: str):
        """
        新しい観測を処理し、未知であれば新しいシンボルを割り当てる。
        """
        if not isinstance(observation, dict):
            return

        obs_hash = self._get_observation_hash(observation)

        if obs_hash not in self.known_concepts:
            # --- 新しい概念（シンボル）の創発 ---
            self.known_concepts.add(obs_hash)
            new_concept_id = f"concept_{self.concept_counter}"
            self.concept_counter += 1
            
            print(f"✨ 新しい概念を発見！ シンボル '{new_concept_id}' を割り当てます。")

            # --- ナレッジグラフへの記録 ---
            # 新しい概念と、それが観測された文脈を関連付ける
            self.rag_system.add_relationship(
                source_concept=new_concept_id,
                relation="was observed during",
                target_concept=context
            )
            # 観測内容そのものも記録
            for key, value in observation.items():
                if isinstance(value, (str, int, float)):
                    self.rag_system.add_relationship(
                        source_concept=new_concept_id,
                        relation=f"has property '{key}'",
                        target_concept=str(value)
                    )