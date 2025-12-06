# test_env.py (MODIFIED for MRS environment test)
from ed_env import EDEmergencyEnv

def test_environment():
    print("Testing ED Environment with Multi-Resource Scheduling (MRS)...")
    
    try:
        # Use the same parameters as training
        env = EDEmergencyEnv("ED_triage.csv", "ED_admission.csv", max_patients=8, episode_length=50)
        obs, info = env.reset()
        print(f"✅ Environment loaded successfully!")
        # --- MODIFICATION: Check flattened observation shape ---
        print(f"   Observation shape: {obs.shape} (Flattened)")
        print(f"   Action space size: {env.action_space.n} (Combinatorial)")
        print(f"   Number of features (per patient): {env.n_features}")
        
        # Test a few steps
        print("\nTesting steps...")
        for i in range(3):
            # The action space is combinatorial: (P*R + 1)
            action = env.action_space.sample() 
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")
            
        print("\n✅ Environment test passed!")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        raise

if __name__ == "__main__":
    test_environment()