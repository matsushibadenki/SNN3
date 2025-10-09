# matsushibadenki/snn3/snn_research/agent/memory.py
#
# Title: 長期記憶システム
#
# 概要：エージェントの経験を構造化データとして記録・管理する。
#
# 改善点:
# - ROADMAPフェーズ2に基づき、因果的記憶アクセスを実装。
# - 報酬が高かった成功体験を優先的に検索する`retrieve_successful_experiences`メソッドを追加。

import json
from datetime import datetime
from typing import List, Dict, Any

class Memory:
    """
    エージェントの経験を構造化されたタプルとして長期記憶に記録するクラス。
    ファイルは追記専用のjsonl形式で保存される。
    """
    def __init__(self, memory_path="runs/agent_memory.jsonl"):
        """
        Args:
            memory_path (str): 記憶を保存するファイルへのパス。
        """
        self.memory_path = memory_path

    def record_experience(self, state, action, result, reward, expert_used, decision_context):
        """
        単一の経験を記録する。
        """
        experience_tuple = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": state,
            "action": action,
            "result": result,
            "reward": reward,
            "expert_used": expert_used,
            "decision_context": decision_context
        }
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experience_tuple, ensure_ascii=False) + "\n")

    def retrieve_similar_experiences(self, query_state, top_k=5) -> List[Dict[str, Any]]:
        """
        現在の状態に類似した過去の経験を検索する（簡易的な実装）。
        実際には、より高度なベクトル検索などが必要になる。
        """
        experiences = []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    experiences.append(json.loads(line))
        except FileNotFoundError:
            return []
        
        # この実装はデモ用。実際には状態をベクトル化し、類似度検索を行う。
        return experiences[-top_k:]

    def retrieve_successful_experiences(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        過去の経験の中から、特に報酬が高かったものを検索する。
        """
        experiences = []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    experiences.append(json.loads(line))
        except FileNotFoundError:
            return []

        # 報酬（外部報酬と物理的報酬の合計）に基づいて経験をソート
        def get_total_reward(exp: Dict[str, Any]) -> float:
            reward_info = exp.get("reward", {})
            if isinstance(reward_info, dict):
                # 新しい形式: {"external": 0.8, "physical": {...}}
                return float(reward_info.get("external", 0.0))
            elif isinstance(reward_info, (int, float)):
                # 古い形式: 0.8
                return float(reward_info)
            return 0.0

        experiences.sort(key=get_total_reward, reverse=True)
        
        return experiences[:top_k]
