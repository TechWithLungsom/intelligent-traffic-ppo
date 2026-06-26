import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from itms_env import ITMSKaggleJunctionEnv

def main():
    model = PPO.load("./saved_models/ppo_itms_kaggle_model")
    
    env = ITMSKaggleJunctionEnv()
    obs, _ = env.reset()
    
    print("\n========================================================")
    print("  LAUNCHING ITMS DEPLOYMENT MODE FOR: NARENGI TINALI JUNCTION")
    print("========================================================\n")
    
    done = False
    step_count = 0
    
    try:
        while not done and step_count < 20:
            step_count += 1
            action, _ = model.predict(obs, deterministic=True)
            
            direction_map = {0: "NORTH (Green)", 1: "SOUTH (Green)", 2: "EAST (Green)", 3: "WEST (Green)"}
            
            print(f"[TIMESTEP {step_count:02d}] AI Choice: Signaling Phase Shift to -> {direction_map[int(action)]}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"       Current Junction Backlog (Net Pressure): {info['junction_pressure']:.2f} units")
            print(f"       Current Queues Matrix: {np.round(info['queue_snapshot'], 1)}")
            print("-" * 55)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nITMS pipeline shut down cleanly by operator command input.")

if __name__ == "__main__":
    main()