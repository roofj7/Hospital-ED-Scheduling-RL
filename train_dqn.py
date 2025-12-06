# train_dqn.py
from ed_env import EDEmergencyEnv
from dqn_agent import DQNAgent
import numpy as np
import torch

def train_dqn():
    env = EDEmergencyEnv("ED_triage.csv", "ED_admission.csv",
                         max_patients=8, episode_length=100)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 500

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(env, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.replay(env)

            state = next_state
            total_reward += reward

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay   # ✅ moved here

        if e % 10 == 0:
            agent.update_target()

        if (e+1) % 50 == 0:
            print(f"Episode {e+1}/{episodes} | Reward={total_reward:.1f} | Epsilon={agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), "dqn_ed_model_mrs_fixed.pth")
    print("\nSaved: dqn_ed_model_mrs_fixed.pth")

if __name__ == "__main__":
    train_dqn()
