"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        self.classes = [None] * 37 # Placeholder to store names in order
        self.class_to_idx = {}

        # 1. Parse the official list.txt to get the exact class mappings
        list_txt_path = os.path.join(data_dir, "annotations", "list.txt")
        with open(list_txt_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    
                    # --- THE HIDDEN FILE FILTER ---
                    if filename.startswith(".") or filename.startswith("._"):
                        continue
                    
                    # Ensure the image, mask, AND xml actually exist before adding
                    if os.path.exists(os.path.join(self.images_dir, f"{filename}.jpg")) and \
                       os.path.exists(os.path.join(self.masks_dir, f"{filename}.png")):
                       
                        breed_name = "_".join(filename.split("_")[:-1])
                        class_id = int(parts[1]) - 1 # Official ID is 1-indexed, we need 0-indexed
                        
                        self.filenames.append(filename)
                        self.class_to_idx[breed_name] = class_id
                        self.classes[class_id] = breed_name

        # 2. THE ANTI-LEAKAGE LOCK: 
        # Sort the filenames alphabetically to guarantee 100% deterministic ordering 
        # before train.py applies its seeded random split.
        self.filenames.sort()

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
        has_bbox = os.path.exists(xml_path)

        if has_bbox:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            xmin = int(root.find('.//xmin').text)
            xmax = int(root.find('.//xmax').text)
            ymin = int(root.find('.//ymin').text)
            ymax = int(root.find('.//ymax').text)

            # 4. Create a min-max Bounding Box
            bbox = [xmin, ymin, xmax, ymax]
        else:
            img_h, img_w = img.shape[:2]
            bbox = [0, 0, img_w, img_h]

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
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)
            img = img / 255.0

        if bbox[0] != 0 or bbox[1] != 0 or bbox[2] != 0 or bbox[3] != 0: # Only convert if the box wasn't cropped out
            bbox = [
                (bbox[0] + bbox[2]) / 2.0,  # x_center
                (bbox[1] + bbox[3]) / 2.0,  # y_center
                (bbox[2] - bbox[0]),        # width
                (bbox[3] - bbox[1])         # height
            ]

        # 6. Adjust the mask and return (image, label, bbox, mask)
        mask = torch.clamp(torch.as_tensor(mask, dtype=torch.long) - 1, min=0, max=2)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return img, label_idx, bbox_tensor, mask, has_bbox
