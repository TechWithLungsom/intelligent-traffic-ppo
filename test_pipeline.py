import pytest
import numpy as np
from itms_env import ITMSKaggleJunctionEnv

def test_kaggle_env_ingestion():
    env = ITMSKaggleJunctionEnv()
    obs, _ = env.reset()
    assert obs.shape == (8,)
    assert isinstance(obs, np.ndarray)
    assert env.current_step == 0
    assert len(env.location_data) == 4

def test_accident_capacity_constraint():
    env = ITMSKaggleJunctionEnv()
    env.reset()
    
    # Manually introduce an accident anomaly check into the evaluation sequence
    env.location_data[0].loc[env.data_idx + 1, 'accident_reported'] = 1
    env.location_data[0].loc[env.data_idx + 1, 'avg_vehicle_speed'] = 40.0
    
    obs, reward, terminated, truncated, info = env.step(0)
    # Expected result mapping math calculation:
    # 40.0 / 2 + 15 = 35 capacity units -> 35 * 0.25 (accident bottleneck scale) = 8.75 limit bounds
    assert env.outgoing_capacities[0] == 8.75