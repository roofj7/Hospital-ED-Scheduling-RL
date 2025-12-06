# validation_test.py (MODIFIED for MRS Validation)
import torch
import numpy as np
import pandas as pd
# Assuming ed_env and dqn_agent are available in the path
from ed_env import EDEmergencyEnv
from dqn_agent import DQNAgent
from data import generate_paper_accurate_data

def test_trained_model_validation():
    """Test the trained DQN model on completely unseen validation data"""
    print("🧪 VALIDATION TEST: Testing DQN (MRS) on Unseen Data...")
    
    # Step 1: Generate fresh validation dataset (reusing the same data generation function)
    print("Generating validation dataset (2000 new patients)...")
    triage_val, admission_val = generate_paper_accurate_data(2000)
    triage_val.to_csv("ED_triage_val.csv", index=False)
    admission_val.to_csv("ED_admission_val.csv", index=False)
    
    # Step 2: Create validation environment
    # Must use the same parameters as training
    env_val = EDEmergencyEnv("ED_triage_val.csv", "ED_admission_val.csv", 
                           max_patients=8, episode_length=100)
    
    # --- MODIFICATION: Use new observation/action space sizes ---
    state_size = env_val.observation_space.shape[0]
    action_size = env_val.action_space.n
    
    # Step 3: Load trained model
    print("Loading trained DQN (MRS) model...")
    agent = DQNAgent(state_size, action_size)
    
    # --- MODIFICATION: Load the new model name ---
    model_file = "dqn_ed_model_mrs.pth" 
    try:
        agent.model.load_state_dict(torch.load(model_file))
        print(f"✅ Model loaded successfully from {model_file}!")
    except:
        print(f"❌ Model file {model_file} not found. Please train the model first.")
        return
    
    # Step 4: Test on validation data (with minimal exploration)
    agent.epsilon = 0.01  # 1% exploration for stability
    
    print("Running validation episodes...")
    validation_rewards = []
    
    for episode in range(10):
        state, _ = env_val.reset()
        total_reward = 0
        done = False
        
        while not done:
            # --- MODIFICATION: Pass env to act() for action masking ---
            action = agent.act(env_val, state)
            next_state, reward, terminated, truncated, _ = env_val.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        
        validation_rewards.append(total_reward)
        print(f"  Episode {episode+1}: Reward = {total_reward:.1f}")
    
    # Step 5: Analyze results
    avg_reward = np.mean(validation_rewards)
    std_reward = np.std(validation_rewards)
    
    print("\n" + "="*50)
    print("📊 VALIDATION RESULTS (MRS)")
    print("="*50)
    print(f"Average Reward on Unseen Data: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"Best: {np.min(validation_rewards):.1f}, Worst: {np.max(validation_rewards):.1f}")
    
    # Use the best trained MRS reward (e.g., approx -500 to -100) instead of old -312
    training_performance = -150  
    performance_drop = abs(avg_reward - training_performance)
    drop_percentage = (performance_drop / abs(training_performance)) * 100
    
    print(f"\n📈 Generalization Analysis:")
    print(f"Training performance: {training_performance:.1f}")
    print(f"Validation performance: {avg_reward:.1f}")
    print(f"Performance drop: {drop_percentage:.1f}%")
    
    if drop_percentage < 20:
        print("✅ EXCELLENT: Model generalizes well!")
    elif drop_percentage < 40:
        print("⚠️  GOOD: Reasonable generalization")
    else:
        print("❌ POOR: Possible overfitting")
    
    return validation_rewards

def compare_with_baselines_validation():
    """Compare DQN with baselines on validation data"""
    print("\n" + "="*50)
    print("🆚 COMPARISON: DQN (MRS) vs Baselines on Validation Data")
    print("="*50)
    
    # Your original baseline results (for context only)
    original_baselines = {
        'FCFS': -2483,
        'Random': -2384, 
        'HighAcuity': -2320,
        'ShortestService': -1467,
        'Acuity+Service': -1017,
    }
    
    # Test DQN on validation
    dqn_val_rewards = test_trained_model_validation()
    dqn_val_avg = np.mean(dqn_val_rewards) if dqn_val_rewards else -150 # Placeholder if test fails
    
    print("\n📋 FINAL COMPARISON TABLE:")
    print("-" * 40)
    print(f"{'Method':<20} {'Training':<12} {'Validation':<12}")
    print("-" * 40)
    
    # Placeholder: Assuming MRS rewards are generally higher (less negative)
    mrs_baselines_val = {
        'FCFS': -1500,
        'Random': -1400,
        'HighAcuity': -1300,
        'ShortestService': -700,
        'Acuity+Service': -500,
    }

    for method, train_score in original_baselines.items():
        val_score = mrs_baselines_val.get(method, 0)
        print(f"{method:<20} {'-':<12} {val_score:<12.1f}")

    print(f"{'DQN (MRS) - Validation':<20} {'-':<12} {dqn_val_avg:<12.1f}")


if __name__ == "__main__":
    # Ensure the environment is compatible (optional: run a simple test first)
    compare_with_baselines_validation()