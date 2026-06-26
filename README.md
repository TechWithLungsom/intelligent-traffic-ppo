# Multi-Objective ITMS Corridor Automation via DCBF Reinforcement Learning

An Intelligent Traffic Management System (ITMS) that models urban junction configurations as a non-linear spatial-temporal graph topology $G=(V,E)$. It leverages a Proximal Policy Optimization (PPO) agent running over custom vector-aligned Gymnasium environments to solve non-greedy throughput metrics while actively preventing heuristic approach starvation.

---

## 📈 Performance Highlights & Live Ingestion
The system features a dual-mode engine capable of switching dynamically between offline historical dataset replays and live satellite tracking feeds.

* **Simulation Performance:** Executes optimized tensor and array state calculations at **2,280+ FPS** on Apple Silicon native CPU thread pools.
* **Production Validation:** Integrates seamlessly with the **Google Maps Directions API** to extract real-time routing delays (`duration_in_traffic` deltas) around high-density junctions like Narengi Tinali, Guwahati, processing live telemetry with zero structural model modifications.

---

## 🧮 Mathematical Formulation & Bounded Fairness
Rather than utilizing short-sighted localized timer boundaries, the network optimization engine targets a multi-objective composite function. It minimizes macro-level network pressure while applying an exponential penalty vector derived from recursive approach-level neglect tracking metrics:

$$\text{Reward}_{\text{Total}}(t) = R_{\text{Base}}(t) - P_{\text{Starvation}}(t)$$

Where the network equilibrium metric $R_{\text{Base}}(t)$ tracks global load balances:
$$R_{\text{Base}}(t) = - \left( \sum_{i \in \text{Lanes}_{\text{in}}} Q_i(t) - \sum_{j \in \text{Lanes}_{\text{out}}} C_j(t) \right)$$

And the proprietary **Dynamic Constraint-Bounded Fairness (DCBF)** cost function expands non-linearly to prevent queue accumulation anomalies under severe downstream bottlenecks:
$$P_{\text{Starvation}}(t) = \sum_{i \in \text{Lanes}_{\text{in}}} \alpha \cdot \left(T_i(t)\right)^\beta$$

*(Locked safety scaling parameters: $\alpha = 5.0$, $\beta = 1.5$, tracking continuous starvation step intervals where an approach queue $Q_i \ge Q_{\text{max}} - 1$).*

---

## 📂 System Architecture Tracking Layout

* **`config.yaml`**: Decoupled registry housing structural hyperparameters, geospatial approach coordinate clusters, and private API access profiles (untracked by Git source boundaries).
* **`itms_env.py`**: Core custom Gymnasium micro-simulator that tracks state arrays, controls phase actions, and calculates composite multi-objective reward topologies.
* **`live_maps_pipeline.py`**: Live data ingestion middleware tracking HTTP response vectors from Google Maps services, parsing live delay ratios smoothly into state spaces.
* **`app_dashboard.py`**: Full-stack web application interface built via Streamlit to monitor queue snapshots, real-time starvation lurches, and automated policy decisions.
* **`train_ppo.py`**: Stable-Baselines3 training loop script driving policy-gradient exploration profiles across model training windows.
* **`evaluate_analytics.py`**: Profiling benchmark layout plotting comparative reward performance against random and pre-timed baselines.
* **`test_pipeline.py`**: Automated unit testing component powered by `pytest` ensuring environmental and state matrix integrity.

---

## 🚀 Execution Requirements Sequence

### 1. Local Virtual Environment Deployment
Activate your isolated environment, verify the test suites pass, and launch the web server locally:
```bash
# Activate your isolated virtual environment container context
source m1_traffic_env/bin/activate

# Execute automated unit test cases
pytest test_pipeline.py

# Launch the interactive full-stack web interface
streamlit run app_dashboard.py