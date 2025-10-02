# matsushibadenki/snn3/run_rl_agent.py
# Title: 強化学習エージェント実行スクリプト
# Description: 生物学的学習則に基づくSNNエージェントを起動し、
#              強化学習のループを実行します。

import torch
import argparse

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.simple_env import SimpleEnvironment
from tqdm import tqdm  # type: ignore

def main():
    parser = argparse.ArgumentParser(description="Biologically Plausible Reinforcement Learning Framework")
    parser.add_argument("--episodes", type=int, default=100, help="Number of learning episodes.")
    parser.add_argument("--pattern_size", type=int, default=10, help="Size of the pattern to match.")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 環境とエージェントを初期化
    env = SimpleEnvironment(pattern_size=args.pattern_size, device=device)
    agent = ReinforcementLearnerAgent(input_size=args.pattern_size, output_size=args.pattern_size, device=device)

    print("\n" + "="*20 + "🤖 生物学的強化学習開始 🤖" + "="*20)
    
    progress_bar = tqdm(range(args.episodes))
    total_reward = 0.0

    for episode in progress_bar:
        state = env.reset()
        
        # 1. 行動選択
        action = agent.get_action(state)
        
        # 2. 環境との相互作用
        next_state, reward, done = env.step(action)
        
        # 3. 学習
        agent.learn(reward)
        
        total_reward += reward
        avg_reward = total_reward / (episode + 1)
        
        progress_bar.set_description(f"Episode {episode+1}/{args.episodes}")
        progress_bar.set_postfix({"Last Reward": f"{reward:.2f}", "Avg Reward": f"{avg_reward:.3f}"})

    print("\n" + "="*20 + "✅ 学習完了" + "="*20)
    print(f"最終的な平均報酬: {avg_reward:.4f}")


if __name__ == "__main__":
    main()

