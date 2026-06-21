import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from itms_env import ITMSKaggleJunctionEnv

def main():
    # 1. Load the trained policy model weights out of serialization memory
    model = PPO.load("./saved_models/ppo_itms_kaggle_model")
    
    # 2. Instantiate a fresh validation environment pass
    env = ITMSKaggleJunctionEnv()
    obs, _ = env.reset()
    
    print("\n========================================================")
    # Highlight your local university junction to match your project pitch context
    print("  LAUNCHING ITMS DEPLOYMENT MODE FOR: NARENGI TINALI JUNCTION")
    print("========================================================\n")
    
    done = False
    step_count = 0
    
    try:
        while not done and step_count < 20:
            step_count += 1
            # Pass the observation tensor to your trained neural network
            action, _ = model.predict(obs, deterministic=True)
            
            # Map discrete scalar predictions back to human-readable direction fields
            direction_map = {0: "NORTH (Green)", 1: "SOUTH (Green)", 2: "EAST (Green)", 3: "WEST (Green)"}
            
            print(f"[TIMESTEP {step_count:02d}] AI Choice: Signaling Phase Shift to -> {direction_map[int(action)]}")
            
            # Read incoming data strings from dataset indices
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"       Current Junction Backlog (Net Pressure): {info['junction_pressure']:.2f} units")
            print(f"       Current Queues Matrix: {np.round(info['queue_snapshot'], 1)}")
            print("-" * 55)
            
            # Pause briefly to mimic a human traffic signal countdown interval
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nITMS pipeline shut down cleanly by operator command input.")

if __name__ == "__main__":
    main()