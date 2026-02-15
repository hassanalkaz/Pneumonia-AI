import io
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Pneumonia Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None
transform = None
classes = ["NORMAL", "PNEUMONIA"]


def load_model():
    """Load the PyTorch model on startup."""
    global model, device, transform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model architecture
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load trained weights
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    print("Model loaded successfully!")


# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


@app.get("/")
async def root():
    return {"message": "Pneumonia Detection API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image and get pneumonia prediction.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Apply transforms
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
        
        # Get probabilities for both classes
        prob_normal = round(probs[0][0].item() * 100, 2)
        prob_pneumonia = round(probs[0][1].item() * 100, 2)
        
        return PredictionResponse(
            prediction=classes[pred.item()],
            confidence=round(confidence.item() * 100, 2),
            probabilities={
                "NORMAL": prob_normal,
                "PNEUMONIA": prob_pneumonia
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)