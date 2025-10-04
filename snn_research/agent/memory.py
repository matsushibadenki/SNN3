# matsushibadenki/snn3/snn_research/agent/memory.py
# 自律エージェントの長期記憶システム
# フェーズ6の要件に基づき、自己内省と文脈に応じた意思決定を可能にするため、検索・分析機能を追加。

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import Counter

class Memory:
    """
    エージェントの行動、思考、経験を記録・検索・分析するための長期記憶システム。
    自己の内省と文脈に応じた意思決定の基盤となる。
    """
    def __init__(self, memory_path: str = "runs/agent_memory.jsonl"):
        """
        メモリシステムを初期化する。

        Args:
            memory_path (str): 記憶を保存するファイルへのパス。
        """
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        # ファイルが存在しない場合は作成する
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                pass  # 空のファイルを作成

    def _read_memory_file(self) -> List[Dict[str, Any]]:
        """メモリファイルからすべてのエントリを読み込む内部メソッド。"""
        if not os.path.exists(self.memory_path) or os.path.getsize(self.memory_path) == 0:
            return []
        with open(self.memory_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def add_experience(self, agent_name: str, task: str, result: str, status: str):
        """
        エージェントの具体的なタスク経験を記録する。

        Args:
            agent_name (str): この経験をしたエージェントの名前。
            task (str): 実行したタスクの説明。
            result (str): タスク実行の結果。
            status (str): タスクの成否 ("SUCCESS", "FAILURE")。
        """
        details = {"task": task, "result": result, "status": status}
        self.add_entry(agent_name, "EXPERIENCE", details)

    def add_entry(self, agent_name: str, event_type: str, details: Dict[str, Any]):
        """
        新しい記憶（イベント）を記録する。

        Args:
            agent_name (str): このイベントを生成したエージェントの名前。
            event_type (str): イベントの種類 (例: "PLAN_GENERATED", "SELF_EVOLUTION_TRIGGERED")。
            details (Dict[str, Any]): イベントに関する詳細情報。
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_name": agent_name,
            "event_type": event_type,
            "details": details
        }
        
        with open(self.memory_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def retrieve_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        最新の記憶をn件取得する。

        Args:
            n (int): 取得する記憶の数。

        Returns:
            List[Dict[str, Any]]: 最新の記憶エントリのリスト。
        """
        all_memories = self._read_memory_file()
        return all_memories[-n:]

    def search(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """
        指定されたキーと値に一致する記憶を検索する。
        トップレベルのキー（'agent_name', 'event_type'）または'details'内のキーで検索可能。

        Args:
            key (str): 検索対象のキー。
            value (Any): 検索する値。

        Returns:
            List[Dict[str, Any]]: 条件に一致した記憶エントリのリスト。
        """
        all_memories = self._read_memory_file()
        results = []
        for entry in all_memories:
            if key in entry and entry[key] == value:
                results.append(entry)
            elif 'details' in entry and key in entry['details'] and entry['details'][key] == value:
                results.append(entry)
        return results
    
    def analyze_performance(self, task_keyword: str) -> Dict[str, Any]:
        """
        特定のタスクに関する過去のパフォーマンスを分析する。
        成功率と失敗率を計算し、文脈に応じた意思決定を支援する。

        Args:
            task_keyword (str): 分析したいタスクを特定するためのキーワード。

        Returns:
            Dict[str, Any]: パフォーマンス分析結果（試行回数、成功数、失敗数、成功率）。
        """
        experiences = self.search("event_type", "EXPERIENCE")
        relevant_experiences = [
            exp for exp in experiences 
            if 'details' in exp and 'task' in exp['details'] and task_keyword in exp['details']['task']
        ]

        if not relevant_experiences:
            return {"total_attempts": 0, "successes": 0, "failures": 0, "success_rate": 0.0}

        successes = sum(1 for exp in relevant_experiences if exp['details'].get('status') == 'SUCCESS')
        failures = len(relevant_experiences) - successes
        success_rate = (successes / len(relevant_experiences)) * 100 if relevant_experiences else 0.0

        return {
            "task_keyword": task_keyword,
            "total_attempts": len(relevant_experiences),
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate
        }

    def get_summary(self) -> Dict[str, int]:
        """
        記録された全記憶の要約（イベントタイプごとの件数）を返す。
        エージェントの自己内省（イントロスペクション）に使用する。

        Returns:
            Dict[str, int]: 各イベントタイプの発生回数。
        """
        all_memories = self._read_memory_file()
        if not all_memories:
            return {}
        event_types = [entry['event_type'] for entry in all_memories]
        return dict(Counter(event_types))
