FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn redis kafka-python httpx

# Shared domain rules — pricing and catalog construction. Built from the repo
# root (see docker-compose.yml) so this package is reachable; the service itself
# lands at /app so `main:app` still resolves.
COPY core/ ./core/
COPY backend/product_service/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
