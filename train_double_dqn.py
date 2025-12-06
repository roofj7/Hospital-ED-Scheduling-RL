# train_double_dqn.py
from ed_env import EDEmergencyEnv
from double_dqn_agent import DoubleDQNAgent
import numpy as np
import torch
import matplotlib.pyplot as plt

def train_double_dqn():
    env = EDEmergencyEnv("ED_triage.csv", "ED_admission.csv", max_patients=8, episode_length=100)

    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n

    agent = DoubleDQNAgent(state_size, action_size)

    episodes = 500
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    rewards = []
    losses = []

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss:
                losses.append(loss)
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Target update
        if (e + 1) % agent.target_update_every == 0:
            agent.update_target_network()

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards.append(total_reward)

        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.1f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

    torch.save(agent.online.state_dict(), "double_dqn_ed_model.pth")
    print("Saved model as double_dqn_ed_model.pth")

    plt.plot(rewards)
    plt.title("Double DQN Rewards")
    plt.show()

    return agent

if __name__ == "__main__":
    train_double_dqn()
