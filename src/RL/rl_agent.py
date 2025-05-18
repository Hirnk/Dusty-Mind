import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from mindustry_env import MindustryEnv

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        map_shape = observation_space["map"].shape  # (4, 500, 500)
        self.n_channels = map_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.zeros(1, *map_shape)
            cnn_output_dim = self.cnn(sample).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_dim + observation_space["core_items"].shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        map_tensor = obs["map"].float()
        core_items_tensor = obs["core_items"].float()

        cnn_out = self.cnn(map_tensor)
        combined = torch.cat((cnn_out, core_items_tensor), dim=1)

        return self.mlp(combined)

# custom policy

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

env = MindustryEnv()
check_env(env, warn=True)

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_mindustry/")

model.learn(total_timesteps=200_000)

model.save("ppo_mindustry_agent")
