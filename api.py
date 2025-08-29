from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from pycocotools.coco import COCO
import os
import io
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Object Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model definition
class ImprovedCNNDetector(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(32, 448),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(32, 448),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        cls_logits = self.classifier(x)
        bbox_pred = self.bbox_regressor(x)
        return cls_logits, bbox_pred

# Configuration
MODEL_PATH = "best_cnn_detector.pth"
COCO_JSON_PATH = "tirupati_data_robo/test/_annotations.coco.json"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global variables
model = None
num_classes = None
cat_id_to_contiguous_id = {}
contiguous_id_to_cat_id = {}
cat_id_to_name = {}

def initialize_model():
    """Initialize model and category mappings"""
    global model, num_classes, cat_id_to_contiguous_id, contiguous_id_to_cat_id, cat_id_to_name
    
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        if not os.path.exists(COCO_JSON_PATH):
            logger.error(f"COCO annotation file not found: {COCO_JSON_PATH}")
            return False
        
        # Load COCO categories
        coco = COCO(COCO_JSON_PATH)
        cats = coco.loadCats(coco.getCatIds())
        
        # Create mappings
        cat_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(cats)}
        contiguous_id_to_cat_id = {i: cat['id'] for i, cat in enumerate(cats)}
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
        
        num_classes = len(cats)
        logger.info(f"Loaded {num_classes} classes from COCO dataset")
        
        # Load model
        model = ImprovedCNNDetector(num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

# Transform
transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    success = initialize_model()
    if not success:
        logger.error("Failed to initialize model on startup")

@app.get("/")
async def root():
    return {
        "message": "Object Detection API", 
        "status": "ready" if model is not None else "model not loaded",
        "classes": num_classes if num_classes else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy"}

@app.get("/classes")
async def get_classes():
    """Get all available classes"""
    if not cat_id_to_name:
        raise HTTPException(status_code=503, detail="Categories not loaded")
    
    class_names = {}
    for contiguous_id, cat_id in contiguous_id_to_cat_id.items():
        class_names[contiguous_id] = cat_id_to_name[cat_id]
    return {"classes": class_names}

@app.post("/detect")
async def detect_object(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Detect object in uploaded image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Store original dimensions
        orig_width, orig_height = image.size
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            pred_cls, pred_box = model(image_tensor)
            
            # Get predicted class
            class_probs = torch.softmax(pred_cls, dim=1)
            confidence, class_idx = torch.max(class_probs, dim=1)
            confidence = confidence.item()
            class_idx = class_idx.item()
            
            # Map to COCO category
            coco_cat_id = contiguous_id_to_cat_id.get(class_idx)
            class_name = cat_id_to_name.get(coco_cat_id, "Unknown") if coco_cat_id else "Unknown"
            
            # Get bounding box
            bbox = pred_box.squeeze().cpu().tolist()
            
            # Scale to original image size
            scaled_bbox = [
                bbox[0] * orig_width,   # x1
                bbox[1] * orig_height,  # y1
                bbox[2] * orig_width,   # x2
                bbox[3] * orig_height   # y2
            ]
        
        return {
            "filename": file.filename,
            "image_size": {"width": orig_width, "height": orig_height},
            "prediction": {
                "class_id": class_idx,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "coordinates": scaled_bbox,
                    "format": "x1_y1_x2_y2"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)