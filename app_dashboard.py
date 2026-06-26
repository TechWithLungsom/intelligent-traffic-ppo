import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from itms_env import ITMSKaggleJunctionEnv
from live_maps_pipeline import LiveMapsTrafficIngestion
import time

st.set_page_config(page_title="ITMS DCBF Dashboard", layout="wide")

st.title("🚦 Coordinated Regional Pressure Release ITMS via PPO Reinforcement Learning")
st.subheader("Real-Time Inference Deployment Loop — Narengi Tinali Junction Topology")
st.markdown("---")

@st.cache_resource
def load_policy():
    return PPO.load("./saved_models/ppo_itms_kaggle_model")

model = load_policy()

# System Controls Sidebar
st.sidebar.header("🕹️ System Deployment Controls")
run_simulation = st.sidebar.button("Launch Inference Loop", type="primary")
delay_slider = st.sidebar.slider("Step Interval (Seconds)", 0.2, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("📡 Ingestion Telemetry Source")
data_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Kaggle Historical Replay", "Live Google Maps API Feed"]
)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    action_metric = st.empty()
with m_col2:
    pressure_metric = st.empty()
with m_col3:
    step_metric = st.empty()
with m_col4:
    safety_metric = st.empty()

graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.markdown("### 📊 Live Channel Density & Ingestion Matrix")
    queue_chart = st.empty()
with graph_col2:
    st.markdown("### ⚠️ Recursive Lane Starvation State Logs")
    starvation_chart = st.empty()

st.markdown("### 📋 System Operations Execution Feed")
log_area = st.empty()

if run_simulation:
    env = ITMSKaggleJunctionEnv()
    obs, _ = env.reset()
    
    if data_mode == "Live Google Maps API Feed":
        api_ingestor = LiveMapsTrafficIngestion()
    
    done = False
    step_count = 0
    logs = []
    
    history_steps = []
    history_queues = []
    history_starvation = []
    
    direction_map = {0: "🟢 NORTH Phase", 1: "🟢 SOUTH Phase", 2: "🟢 EAST Phase", 3: "🟢 WEST Phase"}
    colors_map = ["#3182ce", "#dd6b20", "#319795", "#805ad5"]
    
    while not done and step_count < 30:
        step_count += 1
        
        action, _ = model.predict(obs, deterministic=True)
        act_int = int(action)
        
        if data_mode == "Live Google Maps API Feed":
            live_inflow_queues, live_exit_caps = api_ingestor.fetch_live_traffic_states()
            
            env.incoming_queues = live_inflow_queues
            env.outgoing_capacities = live_exit_caps
            
            clearing_rate = 8.0
            actual_discharge = min(env.incoming_queues[act_int], clearing_rate, env.outgoing_capacities[act_int])
            env.incoming_queues[act_int] -= actual_discharge
            
            for i in range(env.num_roads):
                if env.incoming_queues[i] >= (env.max_capacity - 1) and i != act_int:
                    env.starvation_tracker[i] += 1
                else:
                    env.starvation_tracker[i] = max(0.0, env.starvation_tracker[i] - 1)
            
            info = {
                "junction_pressure": np.sum(env.incoming_queues) - np.sum(env.outgoing_capacities),
                "queue_snapshot": env.incoming_queues.copy(),
                "starvation_profile": env.starvation_tracker.copy()
            }
            obs = env._get_observation()
            
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        action_metric.metric("Active Signal Phase", direction_map[act_int])
        pressure_metric.metric("Net Backlog (Pressure)", f"{info['junction_pressure']:.2f}")
        step_metric.metric("Sequential Timestep", f"{step_count}/30")
        
        peak_starvation = np.max(info['starvation_profile'])
        safety_status = "⚠️ CRITICAL" if peak_starvation > 2 else "✅ NOMINAL"
        safety_metric.metric("Starvation Status", safety_status, delta=f"{peak_starvation:.0f} Max Lag")
        
        history_steps.append(step_count)
        history_queues.append(info['queue_snapshot'])
        history_starvation.append(info['starvation_profile'])
        
        df_q = pd.DataFrame(history_queues, columns=["Lane A (North)", "Lane B (South)", "Lane C (East)", "Lane D (West)"])
        df_q.index = history_steps
        queue_chart.line_chart(df_q)
        
        fig, ax = plt.subplots(figsize=(6, 3.2))
        lanes_labels = ["North", "South", "East", "West"]
        ax.bar(lanes_labels, info['starvation_profile'], color=colors_map, edgecolor='black', alpha=0.8)
        ax.set_ylabel("Consecutive Congested Steps")
        ax.set_ylim(0, 10)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        fig.patch.set_facecolor('#ffffff')
        starvation_chart.pyplot(fig)
        plt.close(fig)
        
        source_label = "LIVE MAPS" if data_mode == "Live Google Maps API Feed" else "KAGGE CSV"
        log_str = f"[{source_label} T-{step_count:02d}] AI Choice: {direction_map[act_int]} | Net Pressure: {info['junction_pressure']:.1f} | Queues: {np.round(info['queue_snapshot'],1)}"
        logs.insert(0, log_str)
        log_area.text("\n".join(logs))
        
        time.sleep(delay_slider)
        
    st.success("Deployment inference execution block successfully tracking terminal intervals.")
else:
    st.info("Awaiting execution pipeline trigger. Toggle 'Launch Inference Loop' on the sidebar panel.")