# base
FROM python:3.9-slim

# environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# directory
WORKDIR /app

# cache
COPY requirements.txt .

# install
RUN pip install --no-cache-dir -r requirements.txt

# server
COPY server.py .

# artifacts
COPY ids_model.keras .
COPY scaler.joblib .
COPY label_encoder.joblib .
COPY model_columns.json .

# port
EXPOSE 8000

# execute
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]