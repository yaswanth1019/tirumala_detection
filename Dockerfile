FROM python:3.9-slim

WORKDIR /app

# Install only essential dependencies (removed problematic libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY best_cnn_detector.pth .

# Create directory and copy annotation file
RUN mkdir -p tirupati_data_robo/test
COPY tirupati_data_robo/test/_annotations.coco.json ./tirupati_data_robo/test/

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]