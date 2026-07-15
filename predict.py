import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.models.model import get_model
import sys

def predict_image(image_path, model_weight_path="saved_weights/resnet18_defect_model.pth"):
    # 1. Set up the device and model (Using your local CPU for this quick test)
    device = torch.device('cpu') 
    model = get_model(num_classes=2)
    
    # 2. Load your trained brain into the model
    try:
        model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
        model.eval() # Lock the model into evaluation mode
    except FileNotFoundError:
        print("❌ Error: Could not find the model weights. Are they inside the saved_weights/ folder?")
        sys.exit(1)

    # 3. Exactly the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Load and process the image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0) # Add a batch dimension
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)

    # 5. Make the prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        
        # Get the highest probability
        confidence, predicted_class = torch.max(probabilities, 1)
        
    classes = ['Negative (No Crack)', 'Positive (Crack Detected)']
    result = classes[predicted_class.item()]
    conf_score = confidence.item() * 100

    print("\n" + "="*40)
    print("🧠 AI INSPECTION REPORT")
    print("="*40)
    print(f"File: {image_path}")
    print(f"Result: {result}")
    print(f"Confidence: {conf_score:.2f}%\n")

if __name__ == "__main__":
    # If a user provides an image path in the terminal, use it. 
    # Otherwise, ask them for one.
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter the path to a concrete image: ")
        
    predict_image(img_path)