import socket
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= Mindustry Environment TCP Server =================

class MindustryAgentEnv:
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None

        # Dimensions from plugin
        self.layers = 5
        self.width = 400
        self.height = 400
        self.observation_size = self.layers * self.width * self.height + 4  # +4 for float copper

    def start_server(self):
        """Start TCP server and accept one client."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Waiting for client to connect at {self.host}:{self.port}...")
        self.conn, addr = self.sock.accept()
        print(f"Client connected from {addr}")

    def recv_exact(self, n):
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket connection lost")
            data += packet
        return data

    def receive_observation(self):
        """Receive the next observation from client."""
        raw_len = self.recv_exact(4)
        (obs_len,) = struct.unpack('<I', raw_len)
        obs_data = self.recv_exact(obs_len)
        return obs_data

    def decode_observation(self, obs_data):
        """Decode raw bytes into numpy arrays: layers + copper"""
        map_bytes = obs_data[:self.layers * self.width * self.height]
        copper_bytes = obs_data[self.layers * self.width * self.height:]

        map_array = np.frombuffer(map_bytes, dtype=np.uint8)
        map_array = map_array.reshape((self.layers, self.width, self.height))

        copper = struct.unpack('<f', copper_bytes)[0]

        return map_array, copper

    def send_action(self, action_type, x, y, rotation):
        """Send an action to client: pack four ints little-endian."""
        action_bytes = struct.pack('<iiii', action_type, x, y, rotation)
        self.conn.sendall(struct.pack('<I', len(action_bytes)))
        self.conn.sendall(action_bytes)

    def close(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()

# ================= PPO Policy Network =================

class PolicyNet(nn.Module):
    def __init__(self, layers=5, width=400, height=400):
        super().__init__()
        self.conv1 = nn.Conv2d(layers, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)

        conv_out_size = (width // 8) * (height // 8) * 64  # after 3 strides of 2

        self.fc1 = nn.Linear(conv_out_size + 1, 256)  # +1 for copper scalar
        self.fc_action = nn.Linear(256, 2)  # drill or conveyor
        self.fc_x = nn.Linear(256, width)
        self.fc_y = nn.Linear(256, height)
        self.fc_rot = nn.Linear(256, 4)

    def forward(self, x, copper):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, copper.unsqueeze(1)], dim=1)
        x = F.relu(self.fc1(x))

        action_logits = self.fc_action(x)
        x_logits = self.fc_x(x)
        y_logits = self.fc_y(x)
        rot_logits = self.fc_rot(x)

        return action_logits, x_logits, y_logits, rot_logits

# ================= PPO Agent for Inference =================

class PPOAgent:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.policy_net = PolicyNet().to(self.device)
        self.policy_net.eval()

    def select_action(self, observation, copper):
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        copper_tensor = torch.tensor([copper], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_logits, x_logits, y_logits, rot_logits = self.policy_net(obs_tensor, copper_tensor)

            action_probs = torch.softmax(action_logits, dim=1)
            x_probs = torch.softmax(x_logits, dim=1)
            y_probs = torch.softmax(y_logits, dim=1)
            rot_probs = torch.softmax(rot_logits, dim=1)

            action_type = torch.multinomial(action_probs, 1).item()
            x = torch.multinomial(x_probs, 1).item()
            y = torch.multinomial(y_probs, 1).item()
            rot = torch.multinomial(rot_probs, 1).item()

        return action_type, x, y, rot

# ================= Main script to run the server and agent =================

if __name__ == "__main__":
    env = MindustryAgentEnv()
    env.start_server()

    agent = PPOAgent(device='cuda')

    try:
        while True:
            obs_data = env.receive_observation()
            map_array, copper = env.decode_observation(obs_data)

            action_type, x, y, rot = agent.select_action(map_array, copper)

            print(f"Action: type={action_type}, x={x}, y={y}, rot={rot}")

            env.send_action(action_type, x, y, rot)

    except ConnectionError:
        print("Connection closed by client.")
    finally:
        env.close()
