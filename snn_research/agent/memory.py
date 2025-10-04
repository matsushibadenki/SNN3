# snn_research/agent/memory.py
# 長期記憶システム
# 概要：エージェントの経験を構造化データとして記録・管理する。
import json
from datetime import datetime

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

        Args:
            state (dict): タスク実行前の状態。
            action (str): エージェントが実行した行動（例: "run_planner", "evolve_model"）。
            result (dict): 行動の結果。
            reward (float): 行動によって得られた報酬。
            expert_used (list): 使用された専門家SNNのIDリスト。
            decision_context (dict): なぜその行動が選択されたかの文脈情報（内部状態など）。
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

    def retrieve_similar_experiences(self, query_state, top_k=5):
        """
        現在の状態に類似した過去の経験を検索する（簡易的な実装）。
        実際には、より高度なベクトル検索などが必要になる。

        Args:
            query_state (dict): 検索クエリとなる現在の状態。
            top_k (int): 取得する類似経験の数。

        Returns:
            list: 類似した経験タプルのリスト。
        """
        # この実装はデモ用。実際には状態をベクトル化し、
        # FaissやAnnoyのようなライブラリで類似度検索を行う。
        experiences = []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    experiences.append(json.loads(line))
        except FileNotFoundError:
            return []
        
        # ここでは単純に最後のk件を返すことで類似検索を模倣する
        return experiences[-top_k:]