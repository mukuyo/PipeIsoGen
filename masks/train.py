import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def target_transform(mask):
    mask = transforms.ToPILImage()(mask)
    mask = transforms.Resize((256, 256))(mask)
    mask = np.array(mask)  # Convert back to NumPy array
    return mask

# Load dataset
dataset = CustomDataset("data/mask/color", "data/mask/masks", transform=transform, target_transform=target_transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load Deeplabv3 model
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to("cuda" if torch.cuda.is_available() else "cpu"), masks.to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = model(images)['out']

        # Ensure masks are resized to match the output size
        output_size = outputs.size()[2:]  # Get the output size
        masks = F.resize(masks, output_size, interpolation=transforms.InterpolationMode.NEAREST)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}")

# Save the model
torch.save(model.state_dict(), "deeplabv3_model.pth")

# Prediction on a new image
def predict_mask(image_path, model):
    # Check if the file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

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

# Path to test image
test_image_path = "data/mask/color/0.jpg"
try:
    predicted_mask = predict_mask(test_image_path, model)
    cv2.imwrite("data/mask/predicted_mask.png", predicted_mask)
except (FileNotFoundError, ValueError) as e:
    print(f"Error: {e}")



