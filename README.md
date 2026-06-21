# Multi-Objective ITMS Corridor Automation via DCBF Reinforcement Learning

An Intelligent Traffic Management System (ITMS) that models urban junction configurations as a non-linear spatial-temporal graph topology $G=(V,E)$. It leverages a Proximal Policy Optimization (PPO) agent running over custom vector-aligned Gymnasium environments to solve non-greedy throughput metrics while actively preventing heuristic approach starvation.

## Mathematical Formulation & Bounded Fairness
Rather than utilizing short-sighted localized timer boundaries, the network optimization engine targets a composite objective function. It mitigates macro-level network pressure while applying an exponential penalty vector derived from recursive approach-level neglect tracking metrics:

$$\text{Reward}_{\text{Total}}(t) = R_{\text{Base}}(t) - P_{\text{Starvation}}(t)$$

Where the network equilibrium metric $R_{\text{Base}}(t)$ tracks global load balances:
$$R_{\text{Base}}(t) = - \left( \sum_{i \in \text{Lanes}_{\text{in}}} Q_i(t) - \sum_{j \in \text{Lanes}_{\text{out}}} C_j(t) \right)$$

And the proprietary **Dynamic Constraint-Bounded Fairness (DCBF)** cost function expands non-linearly to prevent queue accumulation anomalies under severe downstream bottlenecks:
$$P_{\text{Starvation}}(t) = \sum_{i \in \text{Lanes}_{\text{in}}} \alpha \cdot \left(T_i(t)\right)^\beta$$
*(Locked parameters: $\alpha = 5.0$, $\beta = 1.5$, tracking continuous starvation durations).*

## System Architecture Tracking Layout
* **`config.yaml`**: Decoupled environment parameters registry housing structural training hyperparameters and dataset paths.
* **`itms_env.py`**: Custom data-driven Gymnasium micro-simulator that mirrors historical timeseries influx metrics and dynamically applies incident constraints.
* **`train_ppo.py`**: Local, lightweight training script executing at 2,280+ FPS via vector-aligned array operations.
* **`evaluate_analytics.py`**: Profile benchmarking engine tracking AI performance curves directly against standard baseline heuristics.
* **`run_inference_ui.py`**: Real-time production inference pipeline executing state tracking rollouts.

## Execution Requirements Sequence
```bash
source m1_traffic_env/bin/activate
pytest test_pipeline.py
python3 train_ppo.py
python3 evaluate_analytics.py
python3 run_inference_ui.py