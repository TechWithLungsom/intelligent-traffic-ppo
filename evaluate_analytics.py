import yaml
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from itms_env import ITMSKaggleJunctionEnv

plt.switch_backend('Agg')

def evaluate_and_plot():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = ITMSKaggleJunctionEnv(
        csv_path=config['system']['csv_path'],
        num_roads=config['system']['num_roads'], 
        max_capacity=config['system']['max_capacity']
    )
    model = PPO.load(f"{config['paths']['model_output_dir']}ppo_itms_kaggle_model")

    ai_pressures, random_pressures, ai_queue_states = [], [], []

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ai_pressures.append(info['junction_pressure'])
        ai_queue_states.append(info['queue_snapshot'])

    env.reset()
    done = False
    while not done:
        random_action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
        random_pressures.append(info['junction_pressure'])

    steps = np.arange(len(ai_pressures))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(steps, ai_pressures, label="Kaggle-Trained PPO AI Agent", color="#3182ce", linewidth=2.5)
    ax1.plot(steps, random_pressures, label="Static Heuristic Baseline", color="#e53e3e", linestyle=":")
    ax1.set_title("System-Wide Latency & Pressure Mitigation Analysis (Kaggle Dataset)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Net Queue Backlog Density (Pressure Index)")
    ax1.grid(True, linestyle=":")
    ax1.legend()

    ai_queue_states = np.array(ai_queue_states)
    for lane_idx in range(config['system']['num_roads']):
        ax2.plot(steps, ai_queue_states[:, lane_idx], label=f"Lane Matrix {chr(65+lane_idx)}")
    ax2.set_title("Dynamic Spatial Ingestion Balance Performance Profiles", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Sequential Simulation Steps")
    ax2.set_ylabel("Individual Lane Vehicle Volumes")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(config['paths']['analytics_output'], dpi=300)
    print(f"Visual analytics file successfully generated: {config['paths']['analytics_output']}")

if __name__ == "__main__":
    evaluate_and_plot()