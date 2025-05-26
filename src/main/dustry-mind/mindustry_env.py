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
            "map" : spaces.Box(0, 1, (5, 400, 400), dtype=np.uint8),
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
        width = 400
        total_actions = 2 * (width * width)  # 2 types of blocks

        assert 0 <= action < total_actions, f"Invalid action {action}"

        action_type = action // (width * width)
        index = action % (width * width)
        x = index // width
        y = index % width
        rotation = 0  # You can later randomize or evolve this

        data = struct.pack('<4i', action_type, x, y, rotation)
        self._send_bytes(data)

    def _send_bytes(self, data: bytes):
        length = struct.pack('<I', len(data))
        self.client_socket.sendall(length + data)


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

    def _receive_state(self):
        width, height, channels = 400, 400, 5
        map_bytes = width * height * channels  # 1 byte per cell
        total_bytes = map_bytes + 4  # 4 bytes for float32 copper

        data = self._recv_exact(total_bytes)

        # Decode map as uint8
        obs = np.frombuffer(data[:map_bytes], dtype=np.uint8).reshape((channels, width, height))
        obs = obs.astype(np.float32)  # Convert for ML model

        # Decode copper
        copper = struct.unpack('<f', data[map_bytes:])[0]

        return {
            "map": obs,
            "core_items": np.array([copper], dtype=np.int32)
        }

