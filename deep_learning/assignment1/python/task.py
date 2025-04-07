import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import OpenImagesDataset
from image_util import image_transform, target_transform
from model import DeepLabV3Model, cross_entropy_loss, set_seed


# colour mapping for visualization
CLASS_COLOURS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (0, 255, 255),
    6: (255, 0, 255),
}

# global hyperparameters
NUM_WORKERS = max(os.cpu_count() - 1, 1)
BATCH_SIZE = 16


# training and validation Functions
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()

    running_loss = 0.0

    for i, (images, masks) in enumerate(tqdm(dataloader, desc="Training"), start=1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # outputs: [B, 6, H, W] and masks: [B, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 40 == 0:
            print(f"  Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model on the validation set and compute IoU."""
    model.eval()

    val_loss = 0.0
    iou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            ious = compute_iou(outputs, masks, num_classes)
            iou_scores.append(ious)

    avg_loss = val_loss / len(dataloader)
    avg_iou = np.nanmean(iou_scores, axis=0)
    return avg_loss, avg_iou


if __name__ == "__main__":
    # select device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    set_seed(42)

    # data loading and preprocessing
    all_images_df = pd.read_csv("../data/meta/all_images.csv", index_col=0)[
        ["MaskPath", "ImageID", "DisplayName", "data_split"]
    ]

    # prepare training metadata
    train_images_df = all_images_df.query("data_split == 'train'")
    train_images_meta = train_images_df.groupby("ImageID").apply(
        lambda img: [(row.DisplayName, row.MaskPath) for row in img.itertuples()],
        include_groups=False,
    )

    # prepare validation metadata
    val_images_df = all_images_df.query("data_split == 'validation'")
    val_images_meta = val_images_df.groupby("ImageID").apply(
        lambda img: [(row.DisplayName, row.MaskPath) for row in img.itertuples()],
        include_groups=False,
    )

    # mapping between class names and indices (background=0)
    idx_to_class = dict(enumerate(train_images_df.DisplayName.unique(), start=1))
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    num_classes = max(idx_to_class) + 1

    # datasets and dataloaders with performance options
    dataset_root = "../data/"
    train_dataset = OpenImagesDataset(
        root_dir=dataset_root,
        images_meta=train_images_meta,
        class_to_idx=class_to_idx,
        data_split="train",
        transform=image_transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Use built-in option to drop incomplete batches
    )

    val_dataset = OpenImagesDataset(
        root_dir=dataset_root,
        images_meta=val_images_meta,
        class_to_idx=class_to_idx,
        data_split="validation",
        transform=image_transform,
        target_transform=target_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # hyperparameters, model setup
    num_epochs = 5
    learning_rate = 1e-3

    model = DeepLabV3Model(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # add a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # main training loop
    best_val_loss = float("inf")
    model_save_path = "../output/best_segmentation_model.pth"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, cross_entropy_loss, device
        )
        print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {train_loss:.4f}")

        val_loss, avg_iou = validate(model, val_loader, cross_entropy_loss, device)
        print(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch}/{num_epochs}] Avg IoU (per class): {avg_iou}")

        # save the model if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Model checkpoint saved.")

        scheduler.step()

    print("Training complete.")

    # visualization of results
    model.eval()

    # get a batch from the validation loader
    images, masks = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(images)

    batch_masks = masks.cpu().numpy()
    batch_outputs = outputs.cpu().detach().numpy()

    # plot histogram of ground truth masks (ignoring background)
    plt.hist(batch_masks[batch_masks > 0].flatten(), bins=5, range=(0.5, 5.5))
    plt.title("Mask Histogram")
    plt.show()

    # plot histogram of output logits (for classes other than background)
    plt.hist(batch_outputs[:, 1:].flatten(), bins=50)
    plt.title("Output Logits")
    plt.show()

    # plot histogram of predicted classes
    plt.hist(
        np.argmax(batch_outputs, axis=1).flatten(), bins=5, range=(-0.5, 5.5), log=True
    )
    plt.title("Predicted Classes")
    plt.show()

    # visualize a few examples from the batch
    n_examples = min(len(batch_masks), 8)  # up to 8 examples

    for idx in range(n_examples):
        this_mask = batch_masks[idx]
        this_output = batch_outputs[idx]

        plt.figure(figsize=(16, 6))

        # input image
        plt.subplot(1, 3, 1)
        img_np = images[idx].cpu().detach().numpy().transpose(1, 2, 0)
        plt.imshow(img_np)
        plt.title("Input Image")
        plt.axis("off")

        # ground truth mask
        plt.subplot(1, 3, 2)
        mask_img = np.zeros((*this_mask.shape, 3), dtype=np.uint8)
        for class_idx in range(this_mask.max() + 1):
            mask_img[this_mask == class_idx] = CLASS_COLOURS.get(class_idx, (0, 0, 0))
        plt.imshow(mask_img)
        plt.title("Ground Truth Mask")
        plt.axis("off")

        # predicted mask
        plt.subplot(1, 3, 3)
        predicted_mask = np.argmax(this_output, axis=0)
        predicted_mask_img = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
        for class_idx in range(predicted_mask.max() + 1):
            predicted_mask_img[predicted_mask == class_idx] = CLASS_COLOURS.get(
                class_idx, (0, 0, 0)
            )
        plt.imshow(predicted_mask_img)
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.show()
