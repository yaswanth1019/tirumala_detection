import torch
from model import ImprovedCNNDetector, test_loader, device  # Assuming model.py is in the same directory
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

# === Evaluation on Test Set ===
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            if not targets:
                continue

            imgs = imgs.to(device)

            valid_targets = [t for t in targets if t['labels'].size(0) > 0]
            if not valid_targets:
                continue

            labels = torch.tensor([t['labels'][0] for t in valid_targets]).to(device)
            pred_cls, _ = model(imgs[:len(valid_targets)])

            _, predicted = torch.max(pred_cls, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# === Inference on Single Image ===
def test_single_image(model_path, image_path, coco_json_path):
    # Load category names
    coco = COCO(coco_json_path)
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = ImprovedCNNDetector(num_classes=len(cat_id_to_name))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred_cls, pred_box = model(image_tensor)

        class_id = pred_cls.argmax(dim=1).item()
        class_name = cat_id_to_name.get(class_id, "Unknown")
        bbox = pred_box.squeeze().cpu().tolist()  # [x1, y1, x2, y2]

    # Draw result
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1] - 5, class_name, color='red', fontsize=12, weight='bold')
    plt.title(f"Predicted: {class_name}")
    plt.axis('off')
    plt.show()


# === Main ===
if __name__ == "__main__":
    # Load trained model
    num_classes = len(test_loader.dataset.coco.getCatIds())
    model = ImprovedCNNDetector(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_cnn_detector.pth", map_location=device))

    # Evaluate on test set
    evaluate_model(model, test_loader)

    # Test on single image
    test_single_image(
        model_path="best_cnn_detector.pth",
        image_path="C:/Users/Dr Himangshu/Downloads/Prathik/Prathik/tirupati_data_robo/train/images/Image_60_aug6_jpg.rf.4b6f95ea84a4c31cd8e6f658f235c175.jpg",
        coco_json_path="C:/Users/Dr Himangshu/Downloads/Prathik/Prathik/tirupati_data_robo/test/_annotations.coco.json"
    )