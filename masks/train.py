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
dataset = CustomDataset("data/images/rgb", "data/images/masks", transform=transform, target_transform=target_transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load Deeplabv3 model
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000

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
torch.save(model.state_dict(), "data/model/deeplabv3_model.pth")



