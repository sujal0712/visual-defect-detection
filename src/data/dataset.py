import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split 
def get_data_loaders(data_dir="data/raw", batch_size=32, img_size=224):
    """
    Loads the surface crack images, splits them into training and validation sets,
    and returns PyTorch DataLoaders.
    """
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Make all images 224x224
        transforms.ToTensor(),                   # Convert image pixels to PyTorch math numbers
        transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225] ) # Standardize lighting
    ])
    print(f"Scanning for images in: {data_dir}...")
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    classes = full_dataset.classes
    print(f"Found {len(full_dataset)} total images belonging to classes: {classes}")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    #Create the DataLoaders (these feed the images to the model in batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, classes
    
# security check

if __name__ == "__main__":
    train_loader, val_loader, class_names = get_data_loaders()
    print("Data pipeline is ready!")