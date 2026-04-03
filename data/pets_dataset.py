"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
import albumentations.pytorch as ToTensorV2

from torch.utils.data import Dataset
import numpy as np
import torch

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, data_dir: str, transforms: A.Compose = None):
        super().__init__()
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "annotations", "trimaps")
        self.xmls_dir = os.path.join(data_dir, "annotations", "xmls")

        self.transforms = transforms
        self.filenames = []
        self.classes = set()

        # 1. Scan the images directory and filter out the bad files
        for file in os.listdir(self.images_dir):
            if file.startswith("._") or not file.endswith(".jpg"):
                continue    # Skip the non-JPG files and hidden files

            filename_no_ext = os.path.splitext(file)[0]

            if os.path.exists(os.path.join(self.xmls_dir, f"{filename_no_ext}.xml")) and os.path.exists(os.path.join(self.masks_dir, f"{filename_no_ext}.png")):
                self.filenames.append(filename_no_ext)
                breed_name = "_".join(filename_no_ext.split("_")[:-1])
                self.classes.add(breed_name)

        # 2. Create a mapping from breed name to integer (0 to 36)
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {breed: idx for idx, breed in enumerate(self.classes)}

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # 1. Load the RGB image
        img_path = os.path.join(self.images_dir, f"{filename}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        # 2. Load the Mask
        mask_path = os.path.join(self.masks_dir, f"{filename}.png")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        # 3. Load XML and extract xmin, ymin, xmax, ymax
        xml_path = os.path.join(self.xmls_dir, f"{filename}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        xmin = int(root.find('.//xmin').text)
        xmax = int(root.find('.//xmax').text)
        ymin = int(root.find('.//ymin').text)
        ymax = int(root.find('.//ymax').text)

        # 4. Create a min-max Bounding Box
        bbox = [xmin, ymin, xmax, ymax]

        # 5. Apply self.transforms on the image data
        breed_name = "_".join(filename.split("_")[:-1])
        label_idx = self.class_to_idx[breed_name]

        if self.transforms is not None:
            transformed = self.transforms(
                image = img,
                mask = mask,
                bboxes = [bbox],
                class_labels = [label_idx]
            )

            img = transformed['image']
            mask = transformed['mask']
            if(len(transformed['bboxes']) > 0):
                bbox = transformed['bboxes'][0]
            else:
                bbox = [0, 0, 0, 0]

        if bbox != [0, 0, 0, 0]: # Only convert if the box wasn't cropped out
            bbox = [
                (bbox[0] + bbox[2]) / 2.0,  # x_center
                (bbox[1] + bbox[3]) / 2.0,  # y_center
                (bbox[2] - bbox[0]),        # width
                (bbox[3] - bbox[1])         # height
            ]

        # 6. Adjust the mask and return (image, label, bbox, mask)
        mask = torch.as_tensor(mask, dtype=torch.long) - 1
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return img, label_idx, bbox_tensor, mask
