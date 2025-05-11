import json
import socket

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

host = '127.0.0.1'
port = 1337

def fetch(string):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))

        request = {"command": string}
        sock.sendall((json.dumps(request) + '\n').encode())

        with sock.makefile('r') as f:
            response = f.readline()
            data = json.loads(response)
            return np.array(data["map"])

if __name__ == "__main__":
    map_array = fetch("coreResources")

    print("Map shape:", map_array.shape)
    print("Sample values:\n", map_array)

class MapCNNEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=512):
        super(MapCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)  # (500x500 -> 250x250)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)    # (250x250 -> 125x125)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)    # (125x125 -> 63x63)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)   # (63x63 -> 32x32)
        self.fc = nn.Linear(128 * 32 * 32, output_dim)  # Flatten and project to final feature vector

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_encoder = MapCNNEncoder(input_channels=1, output_dim=512)  # You can change input channels based on layers

    def _get_features(self, obs):
        # Preprocess the observation and pass it through the CNN encoder
        return self.cnn_encoder(obs)

    def forward(self, obs, **kwargs):
        # Get features from CNN encoder
        features = self._get_features(obs)

        # Use the features in the actor and critic networks
        return self._predict(features)

    def _predict(self, features, **kwargs):
        # Policy and value estimation
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value