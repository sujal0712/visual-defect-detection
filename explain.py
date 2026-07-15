import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from src.models.model import get_model
import sys
import warnings

# Import the newly installed Grad-CAM tools
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings("ignore")

def explain_image(image_path):
    print("⏳ Powering up the AI X-Ray...")
    
    # 1. Load the Model
    device = torch.device('cpu')
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("saved_weights/resnet18_defect_model.pth", map_location=device, weights_only=True))
    model.eval()

    for param in model.parameters():
        param.requires_grad = True

    # 2. Target the AI's "Visual Cortex"
    # ResNet18's final convolutional layer (where it makes its visual decisions) is layer4
    try:
        target_layers = [model.layer4[-1]]
    except AttributeError:
        print("❌ Error: Could not find layer4. Ensure your get_model() returns a standard ResNet18.")
        sys.exit(1)

    # 3. Prepare the image for the model (Tensors)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        rgb_img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
        
    input_tensor = transform(rgb_img_pil).unsqueeze(0)

    # 4. Prepare the image for drawing the heatmap (Scaled Numpy Array)
    rgb_img_np = np.array(rgb_img_pil.resize((224, 224)), dtype=np.float32) / 255.0

    # 5. Generate the Heatmap
    # We tell the Grad-CAM to explain class '1' (Positive / Crack Detected)
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(1)] 
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # 6. Overlay the red heatmap on the original concrete image
    visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)

    # 7. Save the result
    output_path = "heatmap_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print("\n" + "="*40)
    print(f"✅ SUCCESS! X-Ray saved as: {output_path}")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter the path to a cracked concrete image: ")
        
    explain_image(img_path)