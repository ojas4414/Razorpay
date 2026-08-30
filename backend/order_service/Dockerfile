FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn redis kafka-python httpx

# Shared domain rules — the order validation this service applies is the same
# code path live/engine.py uses. See core/__init__.py.
COPY core/ ./core/
COPY backend/order_service/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
