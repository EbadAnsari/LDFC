import argparse
import os
from collections import Counter
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from resnet50 import resnet50


CLASS_NAMES = {
    0: "Benign",
    1: "Adenocarcinoma",
    2: "Squamous Carcinoma",
}


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load ResNet50 classification model with 3 output classes.
    """
    model = resnet50(num_classes=3, include_top=True)
    model.to(device)

    if weights_path and os.path.isfile(weights_path):
        state_dict = torch.load(weights_path, map_location=device)

        # Allow loading from checkpoints that may wrap the actual state_dict.
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Basic sanity warning: if the classifier head wasn't loaded, predictions
        # will likely be biased or meaningless.
        if missing_keys:
            print(f"WARNING: Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in state_dict: {unexpected_keys}")
        if any(k.startswith("fc.") for k in missing_keys):
            print(
                "WARNING: Classification head weights (fc.*) were not loaded from "
                "the checkpoint. Ensure you are using the correct 3-class ResNet50 "
                "checkpoint; otherwise the model may always predict a single class."
            )
    else:
        raise FileNotFoundError(
            f"Classification weights file not found: {weights_path}. "
            "Provide a valid --weights path to a trained ResNet50 checkpoint."
        )

    model.eval()
    return model


def build_transform(
    input_size: int = 448,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> transforms.Compose:
    """
    Pre-processing for CT scan classification.
    Assumes a single CT slice or projection image.
    """
    # Match training preprocessing in `classification/train.py` by default:
    # Resize to 448x448 and normalize with mean/std = 0.5.
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def load_image(
    image_path: str,
    input_size: int = 448,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load a CT scan image from disk and apply the same preprocessing
    used during training.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size, mean=mean, std=std)
    tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return tensor, img


def predict_single(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 448,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> Tuple[int, str, float]:
    """
    Predict class id (0, 1, 2) and human-readable label for a single CT scan image.
    """
    img_tensor, _ = load_image(image_path, input_size=input_size, mean=mean, std=std)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)  # [1, 3]
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    class_id = int(pred_idx.item())
    class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
    confidence = float(conf.item())

    return class_id, class_name, confidence


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict CT scan class with ResNet50 "
            "(0: Benign, 1: Adenocarcinoma, 2: Squamous Carcinoma). "
            "`--image` can be a single file or a directory."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to CT scan image file OR a directory containing images",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained ResNet50 weights (.pth/.pt) with 3 output classes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g. 'cuda:0' or 'cpu')",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=448,
        help="Input size used during training (default: 448)",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=(0.5, 0.5, 0.5),
        help="Normalization mean (default: 0.5 0.5 0.5)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=(0.5, 0.5, 0.5),
        help="Normalization std (default: 0.5 0.5 0.5)",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Optional path to save per-class count summary CSV (directory mode only)",
    )
    parser.add_argument(
        "--preds-csv",
        type=str,
        default="",
        help="Optional path to save per-image predictions CSV (directory mode only)",
    )

    args = parser.parse_args()

    # Set up device and model once, then reuse.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device=device)

    image_path = args.image

    if os.path.isfile(image_path):
        # Single image
        class_id, class_name, confidence = predict_single(
            image_path=image_path,
            model=model,
            device=device,
            input_size=args.input_size,
            mean=tuple(args.mean),
            std=tuple(args.std),
        )
        print(f"Image            : {image_path}")
        print(f"Predicted class id: {class_id}")
        print(f"Predicted label  : {class_name}")
        print(f"Confidence       : {confidence:.4f}")
    elif os.path.isdir(image_path):
        # Directory of images
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if f.lower().endswith(exts)
        ]

        if not files:
            raise FileNotFoundError(
                f"No image files found in directory: {image_path}"
            )

        # Collect results (do not print per-image by default)
        results = []
        for fpath in sorted(files):
            try:
                class_id, class_name, confidence = predict_single(
                    image_path=fpath,
                    model=model,
                    device=device,
                    input_size=args.input_size,
                    mean=tuple(args.mean),
                    std=tuple(args.std),
                )
                results.append(
                    {
                        "image": fpath,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                    }
                )
            except Exception as e:
                print(f"Failed to process '{fpath}': {e}")

        # Summary counts as a pandas DataFrame
        try:
            import pandas as pd
        except ModuleNotFoundError:
            pd = None

        counts = Counter([r["class_id"] for r in results])
        rows = []
        for cid in sorted(CLASS_NAMES.keys()):
            rows.append(
                {
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                    "count": int(counts.get(cid, 0)),
                }
            )

        if pd is not None:
            df_summary = pd.DataFrame(rows)
            print("\nPrediction summary (counts):")
            print(df_summary.to_string(index=False))
            if args.summary_csv:
                df_summary.to_csv(args.summary_csv, index=False)
                print(f"\nSaved summary CSV to: {args.summary_csv}")
            if args.preds_csv:
                df_preds = pd.DataFrame(results)
                df_preds.to_csv(args.preds_csv, index=False)
                print(f"\nSaved predictions CSV to: {args.preds_csv}")
        else:
            print("\nPrediction summary (counts):")
            for r in rows:
                print(f"{r['class_id']} ({r['class_name']}): {r['count']}")
    else:
        raise FileNotFoundError(
            f"'--image' path does not exist: {image_path}"
        )


if __name__ == "__main__":
    main()

