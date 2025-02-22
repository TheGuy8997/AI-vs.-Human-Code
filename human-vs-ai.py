import os
import socket
import numpy as np
from stable_baselines3 import PPO
from train_ai import KLAXTrafficPatternEnv

MODEL_PATH = "path_to_ai_model_klax.zip"
FG_HOST = "127.0.0.1"
TELNET_PORT = 9000

class TelnetClient:
    """Handles Telnet communication with FlightGear."""
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        """Establish connection to Telnet server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def send(self, command: str):
        """Send a command to the Telnet server."""
        if self.sock:
            self.sock.sendall((command + "\n").encode())

    def receive(self):
        """Receive a response from the Telnet server."""
        if self.sock:
            return self.sock.recv(1024).decode().strip()
        return ""

    def fetch_property(self, property_path: str) -> float:
        """Fetch a property from the FlightGear property tree."""
        self.send(f"get {property_path}")
        response = self.receive()
        try:
            return float(response)
        except ValueError:
            print(f"Failed to parse property {property_path}: {response}")
            return 0.0

    def close(self):
        """Close the Telnet connection."""
        if self.sock:
            self.sock.close()
            self.sock = None

def fetch_human_metrics(client: TelnetClient):
    """Fetch flight performance metrics for the human pilot."""
    print("Human: Perform the traffic pattern. Press Enter when done.")
    input()  # Wait for human to complete their flight

    metrics = {
        "vertical_speed": client.fetch_property("/velocities/vertical-speed-fpm"),
        "g_force": client.fetch_property("/velocities/load-factor"),
        "accuracy": calculate_accuracy(client)  # Replace simulated accuracy
    }
    return metrics

def evaluate_ai(model: PPO, env: KLAXTrafficPatternEnv, client: TelnetClient):
    """Run the AI's flight and evaluate its performance."""
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break

    metrics = {
        "vertical_speed": client.fetch_property("/velocities/vertical-speed-fpm"),
        "g_force": client.fetch_property("/velocities/load-factor"),
        "accuracy": calculate_accuracy(client)  # Replace simulated accuracy
    }
    return metrics

def calculate_accuracy(client: TelnetClient):
    """Calculate flight accuracy by comparing position to ideal traffic pattern."""
    lat = client.fetch_property("/position/latitude-deg")
    lon = client.fetch_property("/position/longitude-deg")

    # Define ideal traffic pattern bounds (KLAX)
    ideal_lat = 33.93
    ideal_lon = -118.40
    distance_error = np.sqrt((lat - ideal_lat) ** 2 + (lon - ideal_lon) ** 2)

    # Normalize accuracy to a score out of 100 (lower error = higher accuracy)
    max_error = 0.05  # Maximum acceptable error (e.g., 5 km deviation)
    accuracy = max(0, 100 - (distance_error / max_error) * 100)
    return accuracy

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please run `train_ai.py` first.")
        exit(1)

    ai_model = PPO.load(MODEL_PATH)
    env = KLAXTrafficPatternEnv()
    client = TelnetClient(FG_HOST, TELNET_PORT)

    try:
        client.connect()

        # Evaluate Human
        human_metrics = fetch_human_metrics(client)
        print(f"Human Performance: {human_metrics}")

        # Evaluate AI
        ai_metrics = evaluate_ai(ai_model, env, client)
        print(f"AI Performance: {ai_metrics}")

        # Compare Scores
        print("\nFlight Rating Comparison:")
        print(f"Human: {human_metrics}")
        print(f"AI: {ai_metrics}")
    finally:
        client.close()
