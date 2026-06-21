import os
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from itms_env import ITMSKaggleJunctionEnv

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_env = ITMSKaggleJunctionEnv(
        csv_path=config['system']['csv_path'],
        num_roads=config['system']['num_roads'], 
        max_capacity=config['system']['max_capacity']
    )
    env = DummyVecEnv([lambda: raw_env])

    compute_device = "cpu"
    print(f"--> Execution Context Engaged Clean Native Thread Pool: [{compute_device.upper()}]")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(config['hyperparameters']['learning_rate']),
        n_steps=config['hyperparameters']['n_steps'],
        batch_size=config['hyperparameters']['batch_size'],
        n_epochs=config['hyperparameters']['n_epochs'],
        gamma=config['hyperparameters']['gamma'],
        verbose=1,
        device=compute_device,
        tensorboard_log=config['paths']['tensorboard_log_dir']
    )

    print(f"Beginning optimization execution loop across {config['hyperparameters']['total_timesteps']} steps...")
    model.learn(total_timesteps=config['hyperparameters']['total_timesteps'])

    os.makedirs(config['paths']['model_output_dir'], exist_ok=True)
    model_save_path = os.path.join(config['paths']['model_output_dir'], "ppo_itms_kaggle_model")
    model.save(model_save_path)
    print(f"Successfully serialized Kaggle-driven model policy metrics at: {model_save_path}")

if __name__ == "__main__":
    main()