import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import networkx as nx

class ITMSKaggleJunctionEnv(gym.Env):
    """
    Patented System Core Architecture: Custom High-Throughput Gymnasium Environment 
    driven natively by Kaggle Time-Series Data with Dynamic Constraint-Bounded Fairness (DCBF).
    Optimized for vector-aligned operations on Apple Silicon architecture.
    """
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, csv_path="smart_traffic_management_dataset.csv", num_roads=4, max_capacity=50):
        super(ITMSKaggleJunctionEnv, self).__init__()
        
        self.num_roads = num_roads
        self.max_capacity = max_capacity
        
        self.action_space = spaces.Discrete(self.num_roads)
        
        low_bound = np.zeros(self.num_roads * 2, dtype=np.float32)
        high_bound = np.concatenate([
            np.full(self.num_roads, self.max_capacity, dtype=np.float32),
            np.full(self.num_roads, self.max_capacity, dtype=np.float32)
        ])
        self.observation_space = spaces.Box(low=low_bound, high=high_bound, dtype=np.float32)
        self.network_graph = nx.star_graph(self.num_roads)
        
        self.df = pd.read_csv(csv_path)
        self.location_data = {}
        for i in range(1, self.num_roads + 1):
            loc_df = self.df[self.df['location_id'] == i].reset_index(drop=True)
            self.location_data[i - 1] = loc_df
            
        self.starvation_tracker = np.zeros(self.num_roads, dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.starvation_tracker = np.zeros(self.num_roads, dtype=np.float32)
        
        max_start = min(len(self.location_data[i]) for i in range(self.num_roads)) - 205
        self.data_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        
        self.incoming_queues = np.zeros(self.num_roads, dtype=np.float32)
        for i in range(self.num_roads):
            volume = self.location_data[i].loc[self.data_idx, 'traffic_volume']
            self.incoming_queues[i] = np.floor(volume / 35.0)
            
        self.outgoing_capacities = np.zeros(self.num_roads, dtype=np.float32)
        for i in range(self.num_roads):
            speed = self.location_data[i].loc[self.data_idx, 'avg_vehicle_speed']
            self.outgoing_capacities[i] = np.floor(speed / 2.0) + 15
            
        return self._get_observation(), {}

    def _get_observation(self):
        return np.concatenate([self.incoming_queues, self.outgoing_capacities]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        self.data_idx += 1
        
        inflow = np.zeros(self.num_roads, dtype=np.float32)
        for i in range(self.num_roads):
            volume = self.location_data[i].loc[self.data_idx, 'traffic_volume']
            inflow[i] = volume / 180.0
        self.incoming_queues = np.minimum(self.incoming_queues + inflow, self.max_capacity)
        
        clearing_rate = 8.0
        green_lane = action
        actual_discharge = min(self.incoming_queues[green_lane], clearing_rate, self.outgoing_capacities[green_lane])
        self.incoming_queues[green_lane] -= actual_discharge
        
        for i in range(self.num_roads):
            if self.incoming_queues[i] >= (self.max_capacity - 1) and i != green_lane:
                self.starvation_tracker[i] += 1
            else:
                self.starvation_tracker[i] = max(0.0, self.starvation_tracker[i] - 1)
        
        for i in range(self.num_roads):
            speed = self.location_data[i].loc[self.data_idx, 'avg_vehicle_speed']
            accident = self.location_data[i].loc[self.data_idx, 'accident_reported']
            
            base_cap = np.floor(speed / 2.0) + 15
            if accident == 1:
                base_cap *= 0.25 
                
            self.outgoing_capacities[i] = max(4.0, base_cap)
        
        total_pressure = np.sum(self.incoming_queues) - np.sum(self.outgoing_capacities)
        base_reward = -float(total_pressure)
        
        starvation_penalty = float(np.sum(5.0 * (self.starvation_tracker ** 1.5)))
        
        reward = base_reward - starvation_penalty
        
        terminated = self.current_step >= 200
        truncated = False
        info = {
            "junction_pressure": total_pressure, 
            "queue_snapshot": self.incoming_queues.copy(),
            "starvation_profile": self.starvation_tracker.copy()
        }
        
        return self._get_observation(), reward, terminated, truncated, info