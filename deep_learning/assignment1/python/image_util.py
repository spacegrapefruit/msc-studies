import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps


IMAGE_SIZE = (224, 224)


# preprocessing functions
def pad_and_scale_image(image, size):
    """
    Pad an image to a square (centered) and resize it.
    """
    old_width, old_height = image.size
    max_dim = max(old_width, old_height)
    delta_w = max_dim - old_width
    delta_h = max_dim - old_height

    # left, top, right, bottom
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )

    # apply padding
    fill = (0, 0, 0) if image.mode == "RGB" else 0
    padded_img = ImageOps.expand(image, border=padding, fill=fill)
    resized_img = padded_img.resize(size, Image.LANCZOS)

    return resized_img


def pad_and_scale_mask(mask: np.array, size):
    """
    Pad a mask to a square (centered) and resize it using nearest neighbor interpolation.
    """
    old_height, old_width = mask.shape
    max_dim = max(old_width, old_height)
    delta_w = max_dim - old_width
    delta_h = max_dim - old_height

    # left, top, right, bottom
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )

    # apply padding
    padded_mask = np.pad(
        mask,
        ((padding[1], padding[3]), (padding[0], padding[2])),
        mode="constant",
        constant_values=0,
    )

    resized_mask = cv2.resize(padded_mask, size, interpolation=cv2.INTER_NEAREST)
    return resized_mask


# utility to compute IoU metric
def compute_iou(outputs, masks, num_classes):
    """
    Compute Intersection over Union (IoU) for non-background classes.
    """
    outputs = torch.argmax(outputs, dim=1)
    ious = []
    # Assuming background is class 0
    for cls in range(1, num_classes):
        intersection = ((outputs == cls) & (masks == cls)).sum().item()
        union = ((outputs == cls) | (masks == cls)).sum().item()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return ious


def image_transform(image, size=IMAGE_SIZE):
    image = pad_and_scale_image(image, size)
    return transforms.ToTensor()(image)


def target_transform(mask, size=IMAGE_SIZE):
    mask = pad_and_scale_mask(mask, size)
    return torch.from_numpy(mask).long()
