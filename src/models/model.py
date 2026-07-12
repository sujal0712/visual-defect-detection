import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    """
    Loads a pre-trained ResNet18 model and modifies the head for 2 classes.
    """
    # 1. Load the pre-trained model
    # 'DEFAULT' means use the latest pre-trained weights from ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. Freeze the early layers (we don't want to change the parts that know how to see edges)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Replace the final layer
    # ResNet's original final layer was designed for 1000 classes (ImageNet).
    # We replace it with a new layer that only outputs 2 (Cracked vs Okay).
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
if __name__ == "__main__":
    model = get_model()
    print("Model architecture loaded successfully!")
    print(model)