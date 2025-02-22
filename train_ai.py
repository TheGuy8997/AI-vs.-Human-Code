# train_ai.py
import os
import requests
import pandas as pd
import numpy as np
from gym import Env, spaces
from stable_baselines3 import PPO
from typing import Tuple

# Constants
KLAX_BOUNDS = {
    "lat_min": 33.91, "lat_max": 33.95,
    "lon_min": -118.42, "lon_max": -118.38
}
ALT_MIN, ALT_MAX = 500, 5000
SPEED_MIN, SPEED_MAX = 100, 300
MODEL_PATH = "ai_model_klax.zip"

# OpenSky Network Data Fetch
def fetch_opensky_klax(username: str, password: str) -> pd.DataFrame:
    """
    Fetches historical traffic pattern data for KLAX using OpenSky API.
    """
    api_url = "https://opensky-network.org/api/states/all"
    response = requests.get(api_url, auth=(username, password))
    if response.status_code == 200:
        data = response.json()
        states = data['states']
        columns = ['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact',
                   'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity',
                   'heading', 'geo_altitude', 'squawk', 'spi', 'position_source']
        df = pd.DataFrame(states, columns=columns)

        # Filter for KLAX bounds
        klax_data = df[
            (df['latitude'] >= KLAX_BOUNDS['lat_min']) &
            (df['latitude'] <= KLAX_BOUNDS['lat_max']) &
            (df['longitude'] >= KLAX_BOUNDS['lon_min']) &
            (df['longitude'] <= KLAX_BOUNDS['lon_max'])
        ]
        return klax_data
    else:
        raise RuntimeError(f"Failed to fetch OpenSky data: {response.status_code}")

# Custom Gym Environment for AI
class KLAXTrafficPatternEnv(Env):
    def __init__(self):
        super(KLAXTrafficPatternEnv, self).__init__()

        # Observation space: lat, lon, alt, speed, heading
        self.observation_space = spaces.Box(
            low=np.array([KLAX_BOUNDS["lat_min"], KLAX_BOUNDS["lon_min"], ALT_MIN, SPEED_MIN, 0]),
            high=np.array([KLAX_BOUNDS["lat_max"], KLAX_BOUNDS["lon_max"], ALT_MAX, SPEED_MAX, 360]),
            dtype=np.float32
        )

        # Action space: delta heading, throttle, climb rate
        self.action_space = spaces.Box(
            low=np.array([-10, -0.5, -500]),  # delta heading, throttle, climb rate
            high=np.array([10, 0.5, 500]),
            dtype=np.float32
        )
        self.state = None
        self.done = False

    def step(self, action: np.ndarray):
        delta_heading, throttle, climb_rate = action

        # Simulated state changes
        lat = np.random.uniform(KLAX_BOUNDS["lat_min"], KLAX_BOUNDS["lat_max"])
        lon = np.random.uniform(KLAX_BOUNDS["lon_min"], KLAX_BOUNDS["lon_max"])
        alt = np.clip(self.state[2] + climb_rate * 0.1, ALT_MIN, ALT_MAX)
        speed = np.clip(self.state[3] + throttle * 5, SPEED_MIN, SPEED_MAX)
        heading = (self.state[4] + delta_heading) % 360

        self.state = np.array([lat, lon, alt, speed, heading], dtype=np.float32)
        reward = self._calculate_reward()
        self.done = self._check_done()
        return self.state, reward, self.done, {}

    def _calculate_reward(self):
        lat, lon, alt, _, _ = self.state
        reward = 0
        if ALT_MIN <= alt <= ALT_MAX:
            reward += 1
        return reward

    def _check_done(self):
        lat, lon, alt, _, _ = self.state
        return not (KLAX_BOUNDS["lat_min"] <= lat <= KLAX_BOUNDS["lat_max"] and
                    KLAX_BOUNDS["lon_min"] <= lon <= KLAX_BOUNDS["lon_max"] and
                    ALT_MIN <= alt <= ALT_MAX)

    def reset(self):
        self.state = np.array([
            np.random.uniform(KLAX_BOUNDS["lat_min"], KLAX_BOUNDS["lat_max"]),
            np.random.uniform(KLAX_BOUNDS["lon_min"], KLAX_BOUNDS["lon_max"]),
            np.random.uniform(ALT_MIN, ALT_MAX),
            np.random.uniform(SPEED_MIN, SPEED_MAX),
            np.random.uniform(0, 360)
        ], dtype=np.float32)
        self.done = False
        return self.state

# Train the AI
def train_agent(env: KLAXTrafficPatternEnv):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model

if __name__ == "__main__":
    try:
        opensky_data = fetch_opensky_klax("your_username", "your_password")
        print(f"Fetched {len(opensky_data)} flight records for KLAX.")
    except RuntimeError as e:
        print(f"Error fetching OpenSky data: {e}")
        opensky_data = None

    env = KLAXTrafficPatternEnv()
    ai_model = train_agent(env)
    ai_model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
