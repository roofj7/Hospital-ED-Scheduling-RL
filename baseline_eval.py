# baseline_eval.py (FINAL - Uses Average Reward for Comparison)

import numpy as np
import matplotlib.pyplot as plt
from ed_env import EDEmergencyEnv, SERVICE_TO_RESOURCE_MAP

# --- ACTION MASK (same logic as DQNAgent) ---
def get_valid_actions(env, state):
    M = env.max_patients
    R_names = env.resource_names
    R_types = len(R_names)
    
    resource_status_start_idx = M * env.n_features
    resource_availability = state[resource_status_start_idx:]
    patient_features_flat = state[:resource_status_start_idx].reshape(M, env.n_features)
    
    valid_actions = []
    for p_idx in range(M):
        for r_idx, r_name in enumerate(R_names):
            action_idx = p_idx * R_types + r_idx
            if resource_availability[r_idx] <= 0:
                continue
            
            for service_col, required_resource in SERVICE_TO_RESOURCE_MAP.items():
                if required_resource == r_name:
                    idx = env.service_indices.get(service_col)
                    if idx != -1 and patient_features_flat[p_idx, idx] > 0:
                        valid_actions.append(action_idx)
                        break

    # Always allow No-Op
    valid_actions.append(env.action_space_size - 1)
    return valid_actions

# --- BASELINE POLICIES ---
def fcfs_policy(env, state):
    valid_actions = get_valid_actions(env, state)
    if len(valid_actions) <= 1:
        return env.action_space_size - 1

    M = env.max_patients
    R_types = len(env.resource_names)
    best_wait = -1
    best_action = env.action_space_size - 1

    for action_idx in valid_actions:
        if action_idx == env.action_space_size - 1: 
            continue
        p_idx = action_idx // R_types
        wait = env.waiting_time[p_idx]
        if wait > best_wait:
            best_wait = wait
            best_action = action_idx
    return best_action

def random_policy(env, state):
    valid = get_valid_actions(env, state)
    return np.random.choice(valid)

def improved_high_acuity_policy(env, state):
    valid = get_valid_actions(env, state)
    if len(valid) <= 1:
        return env.action_space_size - 1

    triage_idx = env.triage_idx
    M = env.max_patients
    R_types = len(env.resource_names)
    pf = state[:M * env.n_features].reshape(M, env.n_features)

    best_acuity = 999
    best_action = env.action_space_size - 1

    for action_idx in valid:
        if action_idx == env.action_space_size - 1:
            continue
        p_idx = action_idx // R_types
        acuity = pf[p_idx, triage_idx]
        if acuity < best_acuity:
            best_acuity = acuity
            best_action = action_idx
    return best_action

def shortest_service_policy(env, state):
    valid = get_valid_actions(env, state)
    if len(valid) <= 1:
        return env.action_space_size - 1

    M = env.max_patients
    R_types = len(env.resource_names)
    pf = state[:M * env.n_features].reshape(M, env.n_features)
    service_idxs = [idx for idx in env.service_indices.values() if idx != -1]

    best_val = 999
    best_action = env.action_space_size - 1

    for action_idx in valid:
        if action_idx == env.action_space_size - 1:
            continue
        p_idx = action_idx // R_types
        total = np.sum(pf[p_idx, service_idxs])
        if total < best_val:
            best_val = total
            best_action = action_idx
    return best_action

def acuity_service_combo_policy(env, state):
    valid = get_valid_actions(env, state)
    if len(valid) <= 1:
        return env.action_space_size - 1

    ACUITY_W = 0.7
    SERVICE_W = 0.3

    M = env.max_patients
    R_types = len(env.resource_names)
    pf = state[:M * env.n_features].reshape(M, env.n_features)
    triage_idx = env.triage_idx
    service_idxs = [idx for idx in env.service_indices.values() if idx != -1]

    best_score = -999
    best_action = env.action_space_size - 1

    for action_idx in valid:
        if action_idx == env.action_space_size - 1:
            continue
        p_idx = action_idx // R_types

        acuity_score = 1.0 - pf[p_idx, triage_idx]
        service_score = 1.0 - np.sum(pf[p_idx, service_idxs])
        score = ACUITY_W * acuity_score + SERVICE_W * service_score

        if score > best_score:
            best_score = score
            best_action = action_idx
    return best_action

# --- MAIN EVALUATION — AVERAGE REWARD ---
def evaluate_baselines_improved(episodes=100):
    print("\nEvaluating Baseline Policies (Average Reward)")
    
    policies = {
        "FCFS": fcfs_policy,
        "Random": random_policy,
        "HighAcuity": improved_high_acuity_policy,
        "ShortestService": shortest_service_policy,
        "Acuity+Service": acuity_service_combo_policy
    }

    results = {}

    for name, policy in policies.items():
        env = EDEmergencyEnv("ED_triage.csv", "ED_admission.csv", max_patients=8, episode_length=100)
        rewards = []

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total = 0
            while not done:
                action = policy(env, state)
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += r
            rewards.append(total)

        results[name] = {"mean": np.mean(rewards), "std": np.std(rewards)}
        print(f"{name:20}: {results[name]['mean']:.2f} ± {results[name]['std']:.2f}")

    plot_baseline_results(results)
    return results

def plot_baseline_results(results):
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, means, yerr=stds, capsize=5, alpha=0.8)

    plt.ylabel("Average Reward per Episode")
    plt.title("Baseline Policy Performance (Average Reward)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("baseline_average_comparison.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    evaluate_baselines_improved()
