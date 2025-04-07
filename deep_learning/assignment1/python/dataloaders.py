import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


# dataset definition
class OpenImagesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        data_split,
        images_meta,
        class_to_idx,
        transform=None,
        target_transform=None,
    ):
        """
        Dataset for Open Images segmentation.
        """
        self.root_dir = root_dir
        self.data_split = data_split
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.image_dir = os.path.join(root_dir, "images", data_split)
        self.mask_dir = os.path.join(root_dir, "masks", data_split)
        self.image_meta = images_meta

    def __len__(self):
        return len(self.image_meta)

    def __getitem__(self, idx):
        # get image ID and its associated mask metadata
        img_id = self.image_meta.index[idx]
        img_meta = self.image_meta.iloc[idx]

        # load input image
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # determine mask shape from first mask
        mask_path_first = os.path.join(self.mask_dir, img_meta[0][1])
        mask_shape = np.array(Image.open(mask_path_first)).shape
        combined_mask = np.zeros(mask_shape, dtype=np.uint8)

        # combine masks from all classes into a single mask
        for class_name, mask_rel_path in img_meta:
            class_idx = self.class_to_idx[class_name]
            mask_full_path = os.path.join(self.mask_dir, mask_rel_path)
            mask = np.array(Image.open(mask_full_path), dtype=bool)
            combined_mask[mask] = class_idx
        mask = combined_mask.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
