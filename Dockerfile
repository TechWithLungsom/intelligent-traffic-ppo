FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY config.yaml itms_env.py train_ppo.py evaluate_analytics.py run_inference_ui.py test_pipeline.py smart_traffic_management_dataset.csv ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gymnasium stable-baselines3 torch matplotlib networkx pytest pandas

CMD ["pytest", "test_pipeline.py"]x