import torch
from src.models.model import get_model

model = get_model(num_classes=2)
model.eval()
# 2. Create a "Dummy" image tensor
# [1 image, 3 color channels (RGB), 224 height, 224 width]
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Pass it through the model
with torch.no_grad():
    output = model(dummy_input)

# 4. Check the results
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

# 5. Sanity Check
if output.shape == torch.Size([1, 2]):
    print("\n✅ Success! The model correctly output 2 values for your 2 classes.")
else:
    print("\n❌ Error! The output shape is unexpected.")