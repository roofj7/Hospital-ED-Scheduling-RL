# final_comparison.py (FINAL VERSION)
import numpy as np
import matplotlib.pyplot as plt

from baseline_eval import evaluate_baselines_improved
from evaluate_dqn import evaluate_dqn
from evaluate_double_dqn import evaluate_double_dqn


def final_comparison(episodes=100):

    print("\n=== Evaluating Baseline Policies ===")
    baseline_results = evaluate_baselines_improved(episodes=episodes)
    # baseline_results = { "FCFS": {"mean": x, "std": y}, ... }

    print("\n=== Evaluating DQN ===")
    dqn_avg = evaluate_dqn(episodes=episodes)

    print("\n=== Evaluating Double DQN ===")
    ddqn_avg = evaluate_double_dqn(episodes=episodes)

    # Prepare plotting data
    methods = list(baseline_results.keys()) + ["DQN", "Double DQN"]
    avg_values = [baseline_results[name]["mean"] for name in baseline_results] + [dqn_avg, ddqn_avg]
    std_values = [baseline_results[name]["std"] for name in baseline_results] + [0, 0]

    print("\n" + "="*60)
    print(" FINAL PERFORMANCE COMPARISON (Average Reward per Episode)")
    print("="*60)
    for m, avg in zip(methods, avg_values):
        print(f"{m:20} : {avg:.2f}")
    print("="*60)

    # Plot
    plt.figure(figsize=(12, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F9A828', '#2ECC71', '#9B59B6']

    bars = plt.bar(methods, avg_values, yerr=std_values, capsize=5,
                   color=colors[:len(methods)], alpha=0.85)

    plt.title("DQN & Double DQN vs Baseline Scheduling Policies\n(Average Reward per Episode)",
              fontsize=15, fontweight='bold')
    plt.ylabel("Average Reward (Higher is Better)", fontsize=13)
    plt.grid(axis='y', alpha=0.3)

    # Add labels above bars
    for bar, val in zip(bars, avg_values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f"{val:.1f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("final_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\nSaved comparison plot as final_model_comparison.png\n")


if __name__ == "__main__":
    final_comparison()
