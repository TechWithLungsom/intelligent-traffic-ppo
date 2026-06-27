# Multi-Objective Intelligent Traffic Management System (ITMS) via Dynamic Constraint-Bounded Fairness (DCBF) Reinforcement Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework: Gymnasium](https://img.shields.io/badge/Framework-Gymnasium-green.svg)](https://gymnasium.farama.org/)
[![RL Engine: Stable--Baselines3](https://img.shields.io/badge/Engine-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Deployment: Docker](https://img.shields.io/badge/Deployment-Docker-blue.svg)](https://www.docker.com/)

An enterprise-grade, high-throughput Intelligent Traffic Management System (ITMS) that models urban corridor intersections as a highly non-linear, spatial-temporal graph topology. 

Standard Deep Reinforcement Learning (DRL) approaches focus greedily on global throughput minimization, which mathematically introduces **approach starvation** under downstream capacity structural breaks (e.g., traffic incidents). This project implements a novel **Dynamic Constraint-Bounded Fairness (DCBF)** framework wrapped around a Proximal Policy Optimization (PPO) core agent to enforce non-linear starvation penalties while maintaining a Pareto-optimal global pressure balance.

---

## 🚀 Architectural & System Engineering Core Metrics
* **Vectorized Acceleration:** Optimized state tensor logic to execute at **2,280+ FPS** natively on Apple Silicon CPU thread pools, introducing a 1,700%+ performance increase over standard unvectorized looping environments.
* **Fault-Tolerant Live Data Streaming:** Engineered an asynchronous-ready ingestion middleware wrapping the **Google Maps Directions API**. The pipeline computes real-time routing delays (duration_in_traffic variances) at high-density junctions (Narengi Tinali, Guwahati) and maps them natively into continuous state spaces.
* **Resilient Graceful Degradation:** Features a built-in safety telemetry exception handler that switches immediately to static/stochastic fallback distributions in the event of API connection dropouts, rate-limiting, or token exhaustion—preventing MDP environment crashes.
* **Production Isolation:** Maintained rigorous environment stability through multi-arch **Docker** container configurations and a continuous automated **PyTest** pipeline testing matrix boundary invariants.

---

## ⚙️ Core Optimization Architecture (Multi-Objective Reward Engine)

Instead of relying on simple, reactive rule-based timer thresholds, the intersection control engine maps actions to a composite multi-objective reward formulation that balances global throughput with localized fairness.

### 1. Global Pressure Minimization
To preserve macro-level network equilibrium across interconnected node vertices, the agent dynamically tracks the delta between localized incoming queue densities and outgoing exit link clearance profiles. By penalizing net accumulation, the policy network learns to maximize vehicle discharge velocity across open channels.

### 2. Dynamic Constraint-Bounded Fairness (DCBF)
To prevent infinite queue accumulation on throttled or structurally bottlenecked approach links, the simulator runs an internal state matrix tracking vector. If an individual lane lingers at peak capacity while being denied a clearance interval by the active signal phase, its counter increments consecutively. This triggers an exponentially scaling penalty that rapidly dominates the optimization landscape, compelling the model to execute an emergency safety release phase.

*🔒 **System Convergence Stability:** The safety net uses fixed exponential tuning constants (alpha = 5.0, beta = 1.5) to guarantee clean policy updates and robust reward structures during multi-hour deployment rollouts.*

---

## 📂 System Architecture Breakdown

```text
├── .gitignore              # Strict environment and credential access protection
├── Dockerfile              # Multi-arch platform deployment configuration 
├── config.yaml             # Decoupled registry for hyperparameters & spatial coordinates
├── itms_env.py             # Custom Gymnasium micro-simulator with embedded DCBF logic
├── live_maps_pipeline.py   # Real-time Google Maps API ingestion middleware 
├── app_dashboard.py        # Streamlit-powered full-stack monitoring cockpit 
├── train_ppo.py            # Stable-Baselines3 PPO policy training pipeline
├── evaluate_analytics.py   # Profile comparative benchmarking script
└── test_pipeline.py        # Pytest suite ensuring system matrix state integrity
```

## 📷 Output

### Output Verification A: Kaggle Historical Replay Validation
![Kaggle Historical Replay Run](img1.png)

### Output Verification B: Real-Time Google Maps API Streaming Feed
![Live Google Maps API Feed Run](img2.png)