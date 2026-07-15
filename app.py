from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
from src.models.model import get_model

# 1. Initialize the Web App
app = FastAPI(title="Structural Defect Detection API", version="1.0")

# 2. Load the AI Model into Memory (Runs once when the server starts)
print("⏳ Booting up AI Engine...")
device = torch.device('cpu')
model = get_model(num_classes=2, pretrained=False)
model.load_state_dict(torch.load("saved_weights/resnet18_defect_model.pth", map_location=device, weights_only=True))
model.eval()
print("✅ AI Engine Online!")

# 3. Define the precise Image Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classes = ['Negative (No Crack)', 'Positive (Crack Detected)']

# 4. Create the main Prediction Endpoint
@app.post("/predict/")
async def predict_defect(file: UploadFile = File(...)):
    # Read the raw image bytes sent over the internet
    image_bytes = await file.read()
    
    # SAFETY NET: Try to open the file as an image. If it fails, return a 400 Error.
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload a valid image (JPEG/PNG)."
        )
    
    # Process the image for the neural network
    input_tensor = transform(image).unsqueeze(0)
    
    # Run the AI inspection
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    result = classes[predicted_class.item()]
    conf_score = confidence.item() * 100
    
    # Return a clean JSON response
    return {
        "filename": file.filename,
        "prediction": result,
        "confidence": f"{conf_score:.2f}%"
    }

# 5. Create a simple health-check endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Defect Detection API! Send a POST request to /predict/"}

if __name__ == "__main__":
    # Start the web server on your local machine
    uvicorn.run(app, host="0.0.0.0", port=8000)