import socket
import struct

import gymnasium as gym
import numpy as np

from gymnasium import spaces


class MindustryEnv(gym.Env):

    #setup socket

    def __init__(self, host='127.0.0.1', port=7777):
        super().__init__()

        self.observation_space = spaces.Dict({
            "map" : spaces.Box(0, 1, (4, 400, 400)),
            "core_items": spaces.Box(low=0, high=4000, shape=(1,), dtype=np.int32),
        })

        # 5 actions + map size
        self.action_space = spaces.Discrete(400 * 400 + 5)

        # start socket server

        self.client_socket = None
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(1)

        print("Awaiting connection...")

        self.client_socket, _ = self.server.accept()

        print("Agent successfully bounded to environment")

    # misc

    def _send_command(self, command_str):
        data = command_str.encode('utf-8')
        length = struct.pack('!I', len(data))

        print(command_str, length)

        self.client_socket.sendall(length + data)

    def _send_action(self, action: int):
        self._send_command(f"ACTION:{action}")

    def _recv_exact(self, num_bytes):
        data = b''
        while len(data) < num_bytes:
            packet = self.client_socket.recv(num_bytes - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken.")
            data += packet
        return data

    # setup environment

    def step(self, action):
        self._send_action(action)
        obs = self._receive_state()
        reward = self._calculate_reward(obs)
        done = False
        return obs, reward, done, False, {}

    def reset(self, seed = None, options = None):
        self._send_command("RESET")
        obs = self._receive_state()
        return obs, {}

    def _calculate_reward(self, obs):
        # Custom logic, e.g., copper at core
        return 0.0

    def _receive_state(sock):
        width, height, channels = 400, 400, 5
        map_bytes = width * height * channels  # 1 byte per cell
        total_bytes = map_bytes + 4  # 4 bytes for float32 copper

        data = sock._recv_exact(total_bytes)

        # Decode map as uint8
        obs = np.frombuffer(data[:map_bytes], dtype=np.uint8).reshape((channels, width, height))
        obs = obs.astype(np.float32)  # Convert for ML model

        # Decode copper
        copper = struct.unpack('<f', data[map_bytes:])[0]

        return {
            "map": obs,          # shape (5, 400, 400)
            "copper": copper         # float
        }

