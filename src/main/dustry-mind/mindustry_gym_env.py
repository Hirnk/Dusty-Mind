import gym
import numpy as np
from gym import spaces

from mindustry_env import MindustryAgentEnv


class MindustryGymEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, host='127.0.0.1', port=7766):
        super().__init__()

        # Low-level TCP environment
        self.env = MindustryAgentEnv(host=host, port=port)
        self.env.start_server()

        self.layers = 6
        self.width = 400
        self.height = 400

        # Observation space: 6 layers of 400x400 + copper scalar
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=1, shape=(self.layers, self.width, self.height), dtype=np.uint8),
            "copper": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
        })

        # Action space:
        # action_type: 0=drill, 1=conveyor (2 possible types)
        # x: 0 to width-1
        # y: 0 to height-1
        # rot: 0 to 3
        self.action_space = spaces.MultiDiscrete([2, self.width, self.height, 4])

        # To track copper delta for reward
        self.last_copper = None

    def reset(self):
        print("Env reset: waiting for initial observation...")
        obs_data = self.env.receive_observation()
        map_array, copper = self.env.decode_observation(obs_data)

        self.last_copper = copper

        observation = {"map": map_array, "copper": copper}
        return observation

    def step(self, action):
        # Unpack action
        action_type, x, y, rot = action

        # Send action to environment
        self.env.send_action(action_type, x, y, rot)

        # Receive new observation
        obs_data = self.env.receive_observation()
        map_array, copper = self.env.decode_observation(obs_data)

        # Calculate reward: delta copper gained since last step
        reward = copper - self.last_copper
        self.last_copper = copper

        # Observation dict
        observation = {"map": map_array, "copper": copper}

        # Done: No episode termination logic yet, keep False
        done = False

        # Info dict (optional)
        info = {"copper": copper}

        return observation, reward, done, info

    def close(self):
        self.env.close()

