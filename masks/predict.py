import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load Deeplabv3 model
model = models.segmentation.deeplabv3_resnet50(pretrained=False)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model weights
model_path = "data/model/deeplabv3_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), strict=False)

# Prediction on a new image
def predict_mask(image_path, model):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Convert color format
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    image_resized = cv2.resize(image, (256, 256))
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_resized).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)['out']
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

    # Resize mask to original image size
    mask_resized = cv2.resize(mask, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)

    return mask_resized

# Directory paths
rgb_dir = "data/08_06/rgb/"
mask_dir = "data/08_06/masks"

# Create the mask directory if it doesn't exist
os.makedirs(mask_dir, exist_ok=True)

# Process each image in the rgb directory
for filename in os.listdir(rgb_dir):
    image_path = os.path.join(rgb_dir, filename)
    try:
        predicted_mask = predict_mask(image_path, model)
        mask_filename = os.path.join(mask_dir, f"{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(mask_filename, predicted_mask)
        print(f"Predicted mask saved at {mask_filename}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error processing {image_path}: {e}")
