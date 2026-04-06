import os
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import argparse

from models.multitask import MultiTaskPerceptionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Task Inference Pipeline")
    
    # Target images
    parser.add_argument("-i", "--images", nargs="+", required=True, help="List of image paths to process")
    parser.add_argument("-rn", "--run_name", type=str, default="final_pipeline_showcase", help="WandB run name")
    
    # Checkpoint paths
    parser.add_argument("-c", "--classifier_path", type=str, default="classifier.pth")
    parser.add_argument("-l", "--localizer_path", type=str, default="localizer.pth")
    parser.add_argument("-u", "--unet_path", type=str, default="unet.pth")
    
    # --- Architectural Arguments (Must match training!) ---
    parser.add_argument("-e_b", "--encoder_backbone", type=str, default="unet", choices=["unet", "classifier", "localizer"], help="Which task's encoder to use as the shared backbone")
    parser.add_argument("-d_p", "--dropout_p", type=float, default=0.5, help="Dropout probability used during training")
    # Using a string to boolean conversion for consistency with your train.py
    parser.add_argument("-u_b", "--use_batchnorm", type=lambda x: (str(x).lower() == 'true'), default=True, help="Toggle BatchNorm")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize W&B for inference logging
    wandb.init(
        project="DA6401_Assignment_02",
        name=args.run_name,
        job_type="inference"
    )

    # 2. Load the Unified Model
    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
        encoder_backbone=args.encoder_backbone,
        dropout_p=args.dropout_p,
        use_batchnorm=args.use_batchnorm
    ).to(device)

    clf_ckpt = torch.load(args.classifier_path, map_location="cpu")
    classes = clf_ckpt.get("classes", None)  # None if old checkpoint without it

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
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"Error: Image path '{image_path}' does not exist. Skipping.")
            continue

        print(f"Processing: {image_path}")

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
        if classes is not None:
            breed_name = classes[predicted_class_idx]
        else:
            breed_name = f"Breed_{predicted_class_idx}"

        class_labels_dict = {predicted_class_idx: breed_name} 

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
                        0: "Foreground (Pet)", 
                        1: "Background", 
                        2: "Not Classified (Boundary)"
                    }
                }
            }
        )

        # 8. Log it to the dashboard!
        file_name = os.path.basename(image_path)
        wandb.log({f"Predictions/{file_name}": interactive_image})
    
    print("All interactive plots successfully logged to W&B!")
    wandb.finish()

if __name__ == "__main__":
    main()