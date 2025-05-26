import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from mindustry_gym_env import MindustryGymEnv

def main():
    env = MindustryGymEnv(host='127.0.0.1', port=7766)

    # Optional: check your environment for errors (comment out if verbose)
    check_env(env, warn=True)

    # Wrap with DummyVecEnv for vectorized env interface
    vec_env = DummyVecEnv([lambda: env])

    # Define PPO model
    model = PPO(
        "MultiInputPolicy",  # since obs is dict with map + copper scalar
        vec_env,
        verbose=1,
        device='cpu',       # or 'cuda' if you have GPU
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    # Train for N timesteps (adjust as needed)
    model.learn(total_timesteps=100_000)

    # Save the trained model
    model.save("mindustry_ppo_model")

    env.close()

if __name__ == "__main__":
    main()
