# Use a highly optimized, clean base runtime layer built for multi-arch architectures
FROM python:3.11-slim

# Enforce a strict working space boundary inside the container layout
WORKDIR /app

# Install basic underlying system utilities required for C-expansion compilation checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your local tracking configurations, dataset profiles, and script files down to the layer
COPY config.yaml itms_env.py train_ppo.py evaluate_analytics.py run_inference_ui.py test_pipeline.py smart_traffic_management_dataset.csv ./

# Execute pip compiling steps directly into the container image framework space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gymnasium stable-baselines3 torch matplotlib networkx pytest pandas

# Instruct the runtime interface to automatically test system dependencies on initialization
CMD ["pytest", "test_pipeline.py"]