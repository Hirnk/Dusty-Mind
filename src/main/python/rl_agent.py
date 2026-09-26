import gymnasium as gym
from gymnasium import spaces

import torch.nn as nn
import torch
import json
import socket

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np

HOST = "127.0.0.1"
PORT = 7766  # Changed to integer and matched to Java plugin port

OBS_CHANNELS = 5
OBS_MAP_HEIGHT = 400
OBS_MAP_WIDTH = 400
OBS_SCALAR_FEATURES = 1  # Reduced to 1 to match the single 'copper_amount' feature parsed

NUM_BUILD_TYPES = 3 # No-op, conveyor, drill

class CustomCNN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim: int = 256):
        super().__init__(observation_space, feature_dim)

        map_shape = observation_space["map"].shape
        scalar_shape = observation_space["scalars"].shape[0]

        self.cnn_map = nn.Sequential(
            nn.Conv2d(map_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_tensor = torch.as_tensor(observation_space["map"].sample()[None]).float()
            n_flatten = self.cnn_map(dummy_tensor).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + scalar_shape, feature_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        map_obs = observations["map"]
        scalar_obs = observations["scalars"]

        cnn_out = self.cnn_map(map_obs)
        fc_in = torch.cat((cnn_out, scalar_obs), dim=1)

        return self.linear(fc_in)


class MindustryEnv(gym.Env):
    def __init__(self, host, port, render_mode=None):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = None
        self.conn = None
        self.addr = None
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=1, shape=(OBS_CHANNELS, OBS_MAP_HEIGHT, OBS_MAP_WIDTH), dtype=np.uint8),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SCALAR_FEATURES,), dtype=np.float32)
        })

        self.action_space = spaces.MultiDiscrete([NUM_BUILD_TYPES, 4])

    def _connect_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.conn, self.addr = self.socket.accept()
        self.conn.settimeout(10.0)
        print(f"Connected by {self.addr}")

    def _dump(self, data):
        message = json.dumps(data) + "\n"
        self.conn.sendall(message.encode('utf-8'))

    def _send_data(self, data):
        try:
            self._dump(data)
        except (socket.error, ConnectionResetError) as e:
            print(f"Socket send error: {e}. Reconnecting...")
            self._reconnect()
            self._dump(data)

    def _receive_data(self):
        buffer = ""
        while "\n" not in buffer:
            try:
                chunk = self.conn.recv(4096).decode('utf-8')
                if not chunk:
                    raise ConnectionResetError("Client disconnected or sent empty data!")
                buffer += chunk
            except socket.timeout:
                print("Socket receive timeout. Waiting for data...")
                continue
            except (socket.error, ConnectionResetError) as e:
                print(f"Socket receive error: {e}: Reconnecting...")
                self._reconnect()
                return None

        data_str, buffer = buffer.split("\n", 1)
        return json.loads(data_str), buffer

    def _reconnect(self):
        print("Attempting to reconnect...")
        if self.conn:
            try:
                self.conn.close()
            except socket.error as e:
                print(f"Error closing old connection: {e}")
        if self.socket:
            try:
                self.socket.close()
            except socket.error as e:
                print(f"Error closing old socket: {e}")
        self._connect_socket()

    def _parse_observation(self, raw_obs_data):
        try:
            map_data = np.array(raw_obs_data["map_pixels"], dtype=np.uint8)
            map_data = map_data.reshape(OBS_CHANNELS, OBS_MAP_HEIGHT, OBS_MAP_WIDTH)

            scalar_data = np.array([
                raw_obs_data["copper_amount"]
            ], dtype=np.float32)

            obs = {
                "map": map_data,
                "scalars": scalar_data
            }

            done = raw_obs_data.get("done", False)
            reward = raw_obs_data.get("reward", 0.0)

            return obs, reward, done, False, {}

        except KeyError as e:
            print(f"Error parsing observation: Missing key {e}. Raw data {raw_obs_data}")
            return self.observation_space.sample(), 0.0, True, False, {"error": f"Missing key: {e}"}
        except Exception as e:
            print(f"Unexpected error parsing observation: {e}. Raw data {raw_obs_data}")
            return self.observation_space.sample(), 0.0, True, False, {"error": f"Parsing error: {e}"}

    def step(self, action):
        build_type = action[0]
        rotation = action[1]
        action_data = {}

        if build_type == 0:
            action_data = { "type": "no_op" }
        elif build_type == 1:
            action_data = { "type": "place_drill", "rotation": rotation }
        elif build_type == 2:
            action_data = { "type": "place_conveyor", "rotation": rotation }
        else:
            print("Warning, unknown build index. Sending no_op")
            action_data = { "type": "no_op" }

        self._send_data({"action": action_data})

        received_data = None
        while received_data is None:
            received_data, _ = self._receive_data()
            if received_data is None:
                print("Failed to receive data, retrying...")

        obs, reward, done, truncated, info = self._parse_observation(received_data)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("Environment reset requested. Sending reset command to the game...")
        self._send_data({"command": "reset"})

        received_data = None
        while received_data is None:
            received_data, _ = self._receive_data()
            if received_data is None:
                print("Failed to receive data, retrying...")

        obs, reward, done, truncated, info = self._parse_observation(received_data)
        return obs, info

    def close(self):
        print("Closing socket connection.")
        if self.conn:
            self.conn.close()
        if self.socket:
            self.socket.close()

if __name__ == "__main__":
    env = DummyVecEnv([lambda: MindustryEnv(HOST, PORT)])

    policy_kwargs = dict(
        feature_extractor_class=CustomCNN,
        feature_extractor_kwargs=dict(feature_dim=256),
    )

    model = PPO("MultiInputPolicy", env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log="./ppo_mindustry_tensorboard/")

    print("PPO Agent created, Starting training...")

    try:
        model.learn(total_timesteps=1000000, log_interval=10, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    finally:
        model.save("ppo_mindustry_agent")
        print("Model saved as ppo_mindustry_agent.zip")
        env.close()
        print("Environment closed.")

    print("Training complete.")