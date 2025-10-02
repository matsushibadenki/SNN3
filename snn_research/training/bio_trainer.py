# snn_research/training/bio_trainer.py
# Title: 生物学的強化学習用トレーナー
# Description: 強化学習のパラダイムに合わせ、エージェントと環境を引数に取るように変更。
#              エピソードベースの学習ループ（行動選択 -> 環境作用 -> 学習）を実装。

import torch
from tqdm import tqdm  # type: ignore
from typing import Dict

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.simple_env import SimpleEnvironment

class BioRLTrainer:
    """生物学的強化学習エージェントのためのトレーナー。"""
    def __init__(self, agent: ReinforcementLearnerAgent, env: SimpleEnvironment):
        self.agent = agent
        self.env = env

    def train(self, num_episodes: int) -> Dict[str, float]:
        """強化学習の学習ループを実行する。"""
        progress_bar = tqdm(range(num_episodes))
        total_reward = 0.0

        for episode in progress_bar:
            state = self.env.reset()
            
            # 1. 行動選択
            action = self.agent.get_action(state)
            
            # 2. 環境との相互作用
            next_state, reward, done = self.env.step(action)
            
            # 3. 学習
            self.agent.learn(reward)
            
            total_reward += reward
            avg_reward = total_reward / (episode + 1)
            
            progress_bar.set_description(f"Bio RL Training Episode {episode+1}/{num_episodes}")
            progress_bar.set_postfix({"Last Reward": f"{reward:.2f}", "Avg Reward": f"{avg_reward:.3f}"})

        final_avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        print(f"Training finished. Final average reward: {final_avg_reward:.4f}")
        return {"final_average_reward": final_avg_reward}

