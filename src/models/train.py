import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import get_data_loaders
from src.models.model import get_model
from tqdm import tqdm

def train_model(num_epochs=5):
    # 1. Setup: CPU/GPU detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # 2. Load Data and Model
    train_loader, val_loader, class_names = get_data_loaders()
    model = get_model(num_classes=len(class_names)).to(device)

    # 3. Loss Function and Optimizer
    # CrossEntropyLoss is standard for classification
    criterion = nn.CrossEntropyLoss()
    # Adam is a smart optimizer that adjusts learning speed automatically
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # tqdm adds the progress bar
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # A. Reset gradients
            optimizer.zero_grad()
            
            # B. Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # C. Backward pass (learning)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Loss: {running_loss / len(train_loader):.4f}")
    # Save the trained weights!
    torch.save(model.state_dict(), "saved_weights/resnet18_defect_model.pth")
    print("\nModel saved successfully to saved_weights/resnet18_defect_model.pth")
    
    print("\nTraining complete! Time to evaluate.")

if __name__ == "__main__":
    train_model(num_epochs=3)