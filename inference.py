import os
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from models.multitask import MultiTaskPerceptionModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize W&B for inference logging
    wandb.init(
        project="DA6401_Assignment_02",
        name="interactive_inference",
        job_type="inference"
    )

    # 2. Load the Unified Model
    model = MultiTaskPerceptionModel(
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/unet.pth"
    ).to(device)

    model.eval() # Critical: Turn off dropout

    # 3. Define the EXACT same validation transforms
    vgg_mean = (0.485, 0.456, 0.406)
    vgg_std = (0.229, 0.224, 0.225)

    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=vgg_mean, std=vgg_std),
        ToTensorV2()
    ])

    # 4. Load and preprocess the image
    image_path = "data/oxford-iiit-pet/images/Abyssinian_1.jpg" # Update this to any test image!
    raw_pil = Image.open(image_path).convert("RGB")
    raw_image = np.array(raw_pil)

    transformed = val_transforms(image=raw_image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # 5. Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        
        class_logits = outputs["classification"]
        bbox_coords = outputs["localization"]
        seg_logits = outputs["segmentation"]

    # 6. Process Outputs
    predicted_class_idx = torch.argmax(class_logits, dim=1).item()
    
    # Process Mask (Shape: [224, 224], Values: 0, 1, 2)
    pred_mask = torch.argmax(seg_logits.squeeze(0), dim=0).cpu().numpy()

    # Process Bounding Box [xmin, ymin, xmax, ymax]
    box = bbox_coords.squeeze(0).cpu().numpy()
    xc, yc, w, h = box[0], box[1], box[2], box[3]
    xmin = float(xc - (w / 2))
    xmax = float(xc + (w / 2))
    ymin = float(yc - (h / 2))
    ymax = float(yc + (h / 2))

    # 7. Prepare the Interactive W&B Payload
    # Resize raw image to 224x224 so the pixels align perfectly with the mask/box
    display_img = np.array(raw_pil.resize((224, 224)))

    # Optional: If you have your breed dictionary, paste it here to show names instead of numbers
    class_labels_dict = {predicted_class_idx: f"Breed_{predicted_class_idx}"} 

    interactive_image = wandb.Image(
        display_img,
        boxes={
            "predictions": {
                "box_data": [
                    {
                        "position": {
                            "minX": xmin,
                            "minY": ymin,
                            "maxX": xmax,
                            "maxY": ymax,
                        },
                        "class_id": predicted_class_idx,
                        "domain": "pixel"
                    }
                ],
                "class_labels": class_labels_dict
            }
        },
        masks={
            "segmentation": {
                "mask_data": pred_mask,
                "class_labels": {
                    0: "Pet Body", 
                    1: "Background", 
                    2: "Pet Boundary"
                }
            }
        }
    )

    # 8. Log it to the dashboard!
    wandb.log({"Multi-Task Predictions": interactive_image})
    print("Interactive plot successfully logged to W&B!")
    
    wandb.finish()

if __name__ == "__main__":
    main()