# evaluate_dqn.py
import torch
import numpy as np
from ed_env import EDEmergencyEnv
from dqn_agent import DQNAgent

def evaluate_dqn(model_path="dqn_ed_model_mrs_fixed.pth", episodes=100):

    env = EDEmergencyEnv("ED_triage.csv", "ED_admission.csv",
                         max_patients=8, episode_length=100)

    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    # Greedy (no exploration during evaluation)
    agent.epsilon = 0.0

    # Load weights safely
    try:
        state_dict = torch.load(model_path, map_location=agent.device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=agent.device)

    agent.model.load_state_dict(state_dict)
    agent.model.eval()

    episode_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(env, state)   # IMPORTANT: pass env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)

    print(f"\nDQN — Average Reward Across {episodes} Episodes: {avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    evaluate_dqn()
