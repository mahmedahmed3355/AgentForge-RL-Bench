FROM python:3.12-slim

WORKDIR /app

COPY data /app/data
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PYTHONPATH=/app/data

CMD ["python", "-c", "from gym_env import GymInspectorEnv; env=GymInspectorEnv(); print(env.reset())"]
