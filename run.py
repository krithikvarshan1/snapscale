"""
KLA Problem Statement: AI-Based Restoration of Degraded Images
Entry Point Script: run.py

Usage:
    python run.py <input-dir> <output-dir>

Description:
    Reads degraded .npy images from <input-dir>, performs 2x Super-Resolution
    and Denoising using RestorationNetV2 with 8x Test-Time Augmentation (TTA),
    and saves the restored 256x256 .npy images to <output-dir>.
"""

import sys
import os
import time
import argparse
import numpy as np
import torch

from model import RestorationNetV2

# ------------------------------------------------------------
# TTA HELPERS FOR MAXIMUM RESTORATION FIDELITY
# ------------------------------------------------------------
def apply_tta_transform(tensor, index):
    rotation = index % 4
    t = tensor
    if index >= 4:
        t = torch.flip(t, dims=(-1,))
    if rotation > 0:
        t = torch.rot90(t, rotation, dims=(-2, -1))
    return t

def invert_tta_transform(tensor, index):
    rotation = index % 4
    t = tensor
    if rotation > 0:
        t = torch.rot90(t, -rotation, dims=(-2, -1))
    if index >= 4:
        t = torch.flip(t, dims=(-1,))
    return t

def predict_with_tta(model, noisy_tensor, tta_count=8):
    if tta_count <= 1:
        with torch.no_grad():
            return torch.clamp(model(noisy_tensor), 0.0, 1.0)

    predictions = []
    for i in range(tta_count):
        transformed = apply_tta_transform(noisy_tensor, i)
        pred = model(transformed)
        pred = invert_tta_transform(pred, i)
        predictions.append(pred)

    avg_pred = torch.stack(predictions, dim=0).mean(dim=0)
    return torch.clamp(avg_pred, 0.0, 1.0)

# ------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="KLA Degraded Image Restoration - Evaluation Script"
    )
    parser.add_argument("input_dir", help="Path to input directory containing noisy .npy files")
    parser.add_argument("output_dir", help="Path to output directory to save restored .npy files")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation for faster inference")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    use_tta = not args.no_tta

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Device Setup (NVIDIA GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Locate Model Checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "models", "best_model_v2_blocks20.pth"),
        os.path.join(script_dir, "models", "best_model.pth"),
        os.path.join(script_dir, "best_model_v2_blocks20.pth"),
        os.path.join(script_dir, "best_model.pth"),
        "models/best_model_v2_blocks20.pth",
        "best_model_v2_blocks20.pth"
    ]

    model_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError("Model weight file not found. Ensure best_model_v2_blocks20.pth is inside the models/ folder.")

    print(f"Loading model checkpoint from: {model_path}")

    # 3. Load Model Weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_features = 64
    num_blocks = 20

    if isinstance(checkpoint, dict):
        ckpt_args = checkpoint.get("args", {})
        num_features = ckpt_args.get("num_features", 64)
        num_blocks = ckpt_args.get("num_blocks", 20)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    model = RestorationNetV2(num_features=num_features, num_blocks=num_blocks).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"RestorationNetV2 initialized ({num_blocks} Blocks, {num_features} Features, {total_params:,} parameters)")

    # 4. Find Input Files
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    if len(input_files) == 0:
        raise RuntimeError(f"No .npy files found in input directory: {input_dir}")

    print(f"Found {len(input_files)} .npy image(s) to process.")
    print(f"Mode: {'8x Test-Time Augmentation (TTA)' if use_tta else 'Standard Forward'}")

    # 5. Process Images
    start_total_time = time.time()
    processed_count = 0

    with torch.no_grad():
        for filename in input_files:
            in_file_path = os.path.join(input_dir, filename)
            out_file_path = os.path.join(output_dir, filename)

            # Load numpy array
            img_np = np.load(in_file_path).astype(np.float32)

            # Standardize shape to (1, 1, H, W)
            if img_np.ndim == 2:
                tensor_in = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
            elif img_np.ndim == 3 and img_np.shape[0] == 1:
                tensor_in = torch.from_numpy(img_np).unsqueeze(0)
            elif img_np.ndim == 3 and img_np.shape[-1] == 1:
                tensor_in = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape {img_np.shape} for file {filename}")

            tensor_in = tensor_in.to(device)

            # Forward inference
            if use_tta:
                output_tensor = predict_with_tta(model, tensor_in, tta_count=8)
            else:
                output_tensor = torch.clamp(model(tensor_in), 0.0, 1.0)

            # Convert to numpy output array (H, W)
            output_np = output_tensor.cpu().numpy().squeeze()

            # Technical Compliance Checks
            # 1. Handle any potential NaN / Inf values
            output_np = np.nan_to_num(output_np, nan=0.0, posinf=1.0, neginf=0.0)

            # 2. Strict clamp to [0.0, 1.0] and float32 type
            output_np = np.clip(output_np, 0.0, 1.0).astype(np.float32)

            # Save restored array with exact matching filename
            np.save(out_file_path, output_np)
            processed_count += 1

            if processed_count % 50 == 0 or processed_count == len(input_files):
                print(f"Processed [{processed_count}/{len(input_files)}] files -> {filename}")

    total_time = time.time() - start_total_time
    avg_ms = (total_time / len(input_files)) * 1000.0

    print("\n" + "=" * 60)
    print("RESTORATION COMPLETE")
    print("=" * 60)
    print(f"  Total Images Processed : {processed_count}")
    print(f"  Input Directory        : {input_dir}")
    print(f"  Output Directory       : {output_dir}")
    print(f"  Target Resolution      : 256 x 256")
    print(f"  Average Time           : {avg_ms:.2f} ms/image")
    print(f"  Status                 : SUCCESS")
    print("=" * 60)

if __name__ == "__main__":
    main()
