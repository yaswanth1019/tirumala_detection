# Object Detection API

A FastAPI-based object detection service using PyTorch and COCO dataset annotations.

## Features

- Object detection using custom trained CNN model
- RESTful API with FastAPI
- Support for image upload and batch processing
- Docker containerized for easy deployment
- Ready for Google Cloud Run deployment

## API Endpoints

- `GET /` - Basic API information
- `GET /health` - Health check endpoint
- `GET /classes` - List all available object classes
- `POST /detect` - Upload image for object detection

## Quick Start with Docker

### Prerequisites
- Docker installed on your machine
- Model file: `best_cnn_detector.pth`
- COCO annotations: `tirupati_data_robo/test/_annotations.coco.json`

### Build and Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/object-detection-api.git
cd object-detection-api

# Build Docker image
docker build -t object-detection-api .

# Run container
docker run -p 8080:8080 object-detection-api
```

The API will be available at `http://localhost:8080`

### Test the API

```bash
# Health check
curl http://localhost:8080/health

# Get available classes
curl http://localhost:8080/classes

# Upload image for detection
curl -X POST "http://localhost:8080/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

## Deploy to Google Cloud Run

### Using Google Cloud Console

1. **Fork/Clone this repository** to your GitHub account

2. **Go to Google Cloud Console** → Cloud Run

3. **Click "Create Service"**

4. **Select "Deploy one revision from a source repository"**

5. **Connect to GitHub** and select your repository

6. **Configure deployment**:
   - Branch: `main` (or your default branch)
   - Build Type: Dockerfile
   - Dockerfile path: `/Dockerfile`

7. **Set service configuration**:
   - Service name: `object-detection-api`
   - Region: Choose your preferred region
   - CPU allocation: 2 CPU
   - Memory: 2 GiB
   - Maximum requests per container: 80
   - Timeout: 300 seconds

8. **Set environment variables** (optional):
   - `MODEL_PATH`: `/app/best_cnn_detector.pth`
   - `COCO_JSON_PATH`: `/app/tirupati_data_robo/test/_annotations.coco.json`

9. **Authentication**: 
   - Allow unauthenticated invocations (for public API)
   - Or configure authentication as needed

10. **Click Deploy**

### Using gcloud CLI (Alternative)

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Deploy from GitHub
gcloud run deploy object-detection-api \
    --source https://github.com/YOUR_USERNAME/YOUR_REPO \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

## File Structure

```
object-detection-api/
├── api.py                              # FastAPI application
├── Dockerfile                          # Docker configuration
├── requirements.txt                    # Python dependencies
├── .dockerignore                       # Docker ignore file
├── README.md                          # This file
├── best_cnn_detector.pth              # Trained model (you need to add this)
└── tirupati_data_robo/
    └── test/
        └── _annotations.coco.json     # COCO annotations (you need to add this)
```

## Required Files

Before deploying, make sure you have these files in your repository:

1. **Model file**: `best_cnn_detector.pth` - Your trained PyTorch model
2. **Annotations**: `tirupati_data_robo/test/_annotations.coco.json` - COCO format annotations

## Example Usage

### Python Client

```python
import requests

# API endpoint
url = "https://your-service-url.run.app/detect"

# Upload image
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    
print(response.json())
```

### JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('https://your-service-url.run.app/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Response Format

```json
{
  "filename": "test_image.jpg",
  "image_size": {
    "width": 640,
    "height": 480
  },
  "prediction": {
    "class_id": 1,
    "class_name": "person",
    "confidence": 0.8754,
    "bbox": {
      "coordinates": [100, 150, 250, 400],
      "format": "x1_y1_x2_y2"
    }
  }
}
```

## Environment Variables

- `PORT`: Server port (default: 8080)
- `MODEL_PATH`: Path to model file (default: best_cnn_detector.pth)
- `COCO_JSON_PATH`: Path to COCO annotations (default: tirupati_data_robo/test/_annotations.coco.json)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.