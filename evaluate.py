import torch
from src.data.dataset import get_data_loaders
from src.models.model import get_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress some standard Matplotlib warnings to keep the terminal clean
warnings.filterwarnings("ignore")

def evaluate():
    print("⏳ Loading validation dataset...")
    # We only need the validation loader for this test
    _, val_loader, class_names = get_data_loaders()
    
    # Setup device (CPU is fine, it will just take a minute or two)
    device = torch.device('cpu') 
    model = get_model(num_classes=2)
    
    print("🧠 Loading trained model weights...")
    model.load_state_dict(torch.load("saved_weights/resnet18_defect_model.pth", map_location=device, weights_only=True))
    model.eval() # Lock the model

    all_preds = []
    all_labels = []

    print("🔍 Inspecting thousands of validation images. Please wait...")
    
    # No gradients needed for testing, saves memory and time
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # 1. Print the Hard Math (Precision, Recall, F1-Score)
    print("\n" + "="*50)
    print("📊 FINAL MODEL CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 2. Draw the Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual True Reality')
    plt.xlabel('What the AI Predicted')
    plt.title('Defect Detection Confusion Matrix')
    
    # Save the graph as an image!
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    print("\n✅ Visual Confusion Matrix saved as 'confusion_matrix.png' in your project folder!")

if __name__ == "__main__":
    evaluate()