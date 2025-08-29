import os
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import torchvision
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ------------ Setting Up Dataset --------------
class CustomCocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

        # Create a mapping from COCO category ID to a contiguous index
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(cats)}
        self.contiguous_id_to_cat_id = {i: cat['id'] for i, cat in enumerate(cats)}

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_dir, path)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            return None, None

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            # Map COCO category ID to contiguous index
            labels.append(self.cat_id_to_contiguous_id[ann['category_id']])

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

        return image, target

    def __len__(self):
        return len(self.img_ids)

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.empty(0), []
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets


transform = v2.Compose([
    v2.PILToTensor(),
    # v2.RandomResizedCrop(size=(256, 256), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomVerticalFlip(p=0.5),
    # v2.RandomGrayscale(p=0.3),
    # v2.RandomInvert(p=0.3),
    # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # v2.RandomRotation(degrees=20),  # <--- add rotation
    # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # v2.AutoAugment(),
])


# Setup for rain, test and validation data
train_dataset = CustomCocoDataset(
    img_dir='C:/Users/yaswa/Downloads/Prathik/tirupati_data_robo/train/images',
    ann_file='C:/Users/yaswa/Downloads/Prathik/tirupati_data_robo/train/_annotations.coco.json',
    transform=transform,
)

valid_dataset = CustomCocoDataset(
    img_dir='C:/Users/yaswa/Downloads/Prathik/tirupati_data_robo/valid/images',
    ann_file='C:/Users/yaswa/Downloads/Prathik/tirupati_data_robo/valid/_annotations.coco.json',
    transform=transform,
)

test_dataset = CustomCocoDataset(
    img_dir='C:/Users/yaswa/Downloads/Prathik/tirupati_data_robo/test/images',
    ann_file='C:/Users/yaswa/Downloads/Prathik/tirupati_data_robo/test/_annotations.coco.json',
    transform=transform,
)

print("\n")
print("Length of training data: ",len(train_dataset))
print("Length of testing data: ",len(test_dataset))
print("Length of validation data: ",len(valid_dataset))


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


for img, labels in train_loader:
  print(img.shape)
  break


class ImprovedCNNDetector(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(32, 448),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(32, 448),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        cls_logits = self.classifier(x)
        bbox_pred = self.bbox_regressor(x)
        return cls_logits, bbox_pred
    
# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.coco.getCatIds())
model = ImprovedCNNDetector(num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()

EPOCHS = 300
best_val_loss = float('inf')

if __name__ == "__main__":
    # ------- Training and Validation Loop -----------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_batches = 0
        correct = 0
        total = 0

        for imgs, targets in train_loader:
            # Filter valid images/targets (at least one annotation)
            batch_data = [
                (img, t['labels'][0], t['boxes'][0])
                for img, t in zip(imgs, targets)
                if t['labels'].size(0) > 0
            ]
            if not batch_data:
                continue

            batch_imgs, batch_labels, batch_boxes = zip(*batch_data)
            batch_imgs = torch.stack(batch_imgs).to(device)
            batch_labels = torch.tensor(batch_labels).to(device)
            batch_boxes = torch.stack(batch_boxes).to(device)

            optimizer.zero_grad()
            pred_cls, pred_box = model(batch_imgs)
            loss = criterion_cls(pred_cls, batch_labels) + criterion_bbox(pred_box, batch_boxes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

            # Classification accuracy
            preds = pred_cls.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        avg_train_loss = train_loss / max(train_batches, 1)
        train_acc = 100 * correct / total if total > 0 else 0

        # --------- Validation ---------
        model.eval()
        val_loss = 0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, targets in valid_loader:
                batch_data = [
                    (img, t['labels'][0], t['boxes'][0])
                    for img, t in zip(imgs, targets)
                    if t['labels'].size(0) > 0
                ]
                if not batch_data:
                    continue
                batch_imgs, batch_labels, batch_boxes = zip(*batch_data)
                batch_imgs = torch.stack(batch_imgs).to(device)
                batch_labels = torch.tensor(batch_labels).to(device)
                batch_boxes = torch.stack(batch_boxes).to(device)

                pred_cls, pred_box = model(batch_imgs)
                loss = criterion_cls(pred_cls, batch_labels) + criterion_bbox(pred_box, batch_boxes)
                val_loss += loss.item()
                val_batches += 1

                # Classification accuracy
                preds = pred_cls.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

        avg_val_loss = val_loss / max(val_batches, 1)
        val_acc = 100 * correct / total if total > 0 else 0

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_cnn_detector.pth")
            print(f"Best model saved at epoch {epoch+1}")

        