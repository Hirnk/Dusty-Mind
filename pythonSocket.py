import socket
import json

HOST = '127.0.0.1'  # or the server's IP address
PORT = 1337         # should match the port in your plugin

def send_command(command_dict):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        
        # Send command as JSON string with newline
        message = json.dumps(command_dict) + '\n'
        sock.sendall(message.encode())

        # Receive and decode the response
        response = sock.recv(4096).decode()
        return json.loads(response)

# Test sending a "place block" command
if __name__ == "__main__":
    command = {
        "command": "place",
        "x": 10,
        "y": 15,
        "block": "conveyor"
    }

       
    response = send_command({"command": "get_blocks"})
    print("Response from server:", response)
