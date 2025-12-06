# Emergency Department Scheduling using Deep Reinforcement Learning

This project uses Deep Reinforcement Learning (DQN and Double-DQN) to optimize hospital emergency department workflows by reducing patient waiting time and improving resource utilization.
A custom OpenAI Gym–compatible environment was developed to simulate multi-resource scheduling (doctors, nurses, lab, CT, X-ray) using realistic triage and admission patterns.

# Key Features

Custom Gym environment modeling real hospital treatment workflow
Synthetic dataset (~11,000 samples) based on published ED research
Built-in scheduling baselines:
 * FCFS
 * Acuity-based
 * Shortest-service
RL models trained using DQN and Double-DQN
Reward benchmarking + performance comparison plots

# Technologies Used

Python
PyTorch
NumPy
Matplotlib
Reinforcement Learning (DQN, Double-DQN)
Custom Gym Environment

# Project Structure 
├─ ed_env.py                  # Custom OpenAI Gym environment
├─ dqn_agent.py               # Base DQN implementation
├─ double_dqn_agent.py        # Double-DQN implementation
├─ train_dqn.py               # Training script (DQN)
├─ train_double_dqn.py        # Training script (Double-DQN)
├─ evaluate_dqn.py            # Evaluation script (DQN)
├─ evaluate_double_dqn.py     # Evaluation script (Double-DQN)
├─ final_comparison.py        # Baselines vs RL model results
└─ validation_test.py         # Testing on unseen dataset

# Results Summary

RL models outperformed baseline scheduling strategies in terms of reward and patient waiting time.
Double-DQN provided more stable convergence compared to DQN.
(Training logs and plots can be added in future releases.)
