
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    predict_server:app
