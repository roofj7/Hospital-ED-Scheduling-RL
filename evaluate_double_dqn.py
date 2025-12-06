# evaluate_double_dqn.py
import torch
import numpy as np
from ed_env import EDEmergencyEnv
from double_dqn_agent import DoubleDQNAgent

def evaluate_double_dqn(model_path="double_dqn_ed_model.pth", episodes=100):

    env = EDEmergencyEnv("ED_triage.csv", "ED_admission.csv",
                         max_patients=8, episode_length=100)

    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n

    agent = DoubleDQNAgent(state_size, action_size)

    # Greedy evaluation
    agent_epsilon = 0.0

    try:
        state_dict = torch.load(model_path, map_location=agent.device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=agent.device)

    agent.online.load_state_dict(state_dict)
    agent.online.eval()

    episode_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state, agent_epsilon)  # ACT WITH ENV + EPS
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)

    print(f"\nDouble DQN — Average Reward Across {episodes} Episodes: {avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    evaluate_double_dqn()
