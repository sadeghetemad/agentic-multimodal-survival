FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY agent ./agent
COPY config ./config
COPY services ./services
COPY tools ./tools
COPY app/agentcore_app.py ./app/

EXPOSE 8080

CMD ["python", "-u", "-m", "app.agentcore_app"]