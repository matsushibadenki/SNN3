# matsushibadenki/snn3/run_rl_agent.py
# Title: 強化学習エージェント実行スクリプト
# Description: 生物学的学習則に基づくSNNエージェントを起動し、
#              強化学習のループを実行します。
# 改善点:
# - ROADMAPフェーズ2検証のため、GridWorldEnvに対応。
# - エピソードベースの学習ループを実装し、複数ステップのタスクを実行できるようにした。

import torch
import argparse

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.grid_world import GridWorldEnv
from tqdm import tqdm  # type: ignore

def main():
    parser = argparse.ArgumentParser(description="Biologically Plausible Reinforcement Learning Framework")
    parser.add_argument("--episodes", type=int, default=500, help="Number of learning episodes.")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid world.")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per episode.")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 環境とエージェントを初期化
    env = GridWorldEnv(size=args.grid_size, max_steps=args.max_steps, device=device)
    # 状態ベクトルは4次元(agent_x, agent_y, goal_x, goal_y)、行動は4次元(上/下/左/右)
    agent = ReinforcementLearnerAgent(input_size=4, output_size=4, device=device)

    print("\n" + "="*20 + "🤖 生物学的強化学習開始 (Grid World) 🤖" + "="*20)
    
    progress_bar = tqdm(range(args.episodes))
    total_rewards = []

    for episode in progress_bar:
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 1. 行動選択
            action = agent.get_action(state)
            
            # 2. 環境との相互作用
            next_state, reward, done = env.step(action)
            
            # 3. 学習（毎ステップ報酬を渡すが、実際の重み更新はエピソード終了時に行われる）
            agent.learn(reward)
            
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        # 最近10エピソードの平均報酬を計算
        avg_reward = sum(total_rewards[-10:]) / len(total_rewards[-10:])
        
        progress_bar.set_description(f"Episode {episode+1}/{args.episodes}")
        progress_bar.set_postfix({"Last Reward": f"{episode_reward:.2f}", "Avg Reward (last 10)": f"{avg_reward:.3f}"})

    final_avg_reward = sum(total_rewards) / args.episodes if args.episodes > 0 else 0
    print("\n" + "="*20 + "✅ 学習完了" + "="*20)
    print(f"最終的な平均報酬: {final_avg_reward:.4f}")


if __name__ == "__main__":
    main()
