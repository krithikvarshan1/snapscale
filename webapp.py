"""
Web Application for Semiconductor Image Super-Resolution Viewer.

Enter an image number to see:
  - Noisy Low-Resolution input (128x128)
  - Model Super-Resolved output (256x256)
  - Ground Truth (if available, for train/val images)
  - PSNR / SSIM metrics (when GT is available)
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

import lpips

from losses import ssim
from model import RestorationNetV2

# ============================================================
# CLI ARGUMENT PARSING
# ============================================================

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="SemiSR Interactive Web Viewer & Inference Server"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing noisy .npy files (required)"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory to save restored .npy files (optional)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run web app server on (default: 5000)"
    )
    args, _ = parser.parse_known_args()
    return args

cli_args = parse_cli_args()

CLI_INPUT_DIR = cli_args.input_dir
CLI_OUTPUT_DIR = cli_args.output_dir or "restored_test_outputs"

if not os.path.isdir(CLI_INPUT_DIR):
    print(f"ERROR: Input directory does not exist: {CLI_INPUT_DIR}")
    sys.exit(1)

if CLI_OUTPUT_DIR:
    os.makedirs(CLI_OUTPUT_DIR, exist_ok=True)

# ============================================================
# PATHS & SETUP
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_MODEL_PATHS = [
    os.path.join(SCRIPT_DIR, "models", "best_model_v2_blocks20.pth"),
    os.path.join(SCRIPT_DIR, "best_model_v2_blocks20.pth"),
    "models/best_model_v2_blocks20.pth",
    "best_model_v2_blocks20.pth",
]

MODEL_PATH = None
for path in CANDIDATE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "best_model_v2_blocks20.pth")

SPLIT_FILE = os.path.join(SCRIPT_DIR, "split.json")

# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

# ============================================================
# LOAD MODEL ONCE AT STARTUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

model = RestorationNetV2(
    num_features=ckpt_args.get("num_features", 64),
    num_blocks=ckpt_args.get("num_blocks", 20),
).to(device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print(f"Model loaded on {device}")
print(
    f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
)

# ============================================================
# LOAD LPIPS MODEL ONCE AT STARTUP
# ============================================================

lpips_model = lpips.LPIPS(net="alex").to(device)
lpips_model.eval()
print("LPIPS (AlexNet) metric loaded.")

# ============================================================
# LOAD SPLIT DATA
# ============================================================

split_data = {}
if os.path.exists(SPLIT_FILE):
    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

train_set = set(split_data.get("train", []))
val_set = set(split_data.get("validation", []))
test_set = set(split_data.get("test", []))

# ============================================================
# HELPERS
# ============================================================


def numpy_to_base64_png(array, colormap="gray"):
    """Convert a 2D numpy array to a base64-encoded PNG string."""
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255).astype(np.uint8)
    img = Image.fromarray(array, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def calculate_psnr(prediction, target):
    """Calculate PSNR between two tensors."""
    prediction = torch.clamp(prediction, 0.0, 1.0)
    mse = torch.mean((prediction - target) ** 2)
    if mse.item() == 0:
        return 100.0
    return (10.0 * torch.log10(1.0 / mse)).item()


def calculate_lpips(prediction, target):
    """Calculate LPIPS perceptual distance between two tensors."""
    prediction = torch.clamp(prediction, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    pred_3ch = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
    target_3ch = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
    return lpips_model(pred_3ch, target_3ch).item()


# ============================================================
# TTA HELPERS
# ============================================================


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


def predict_tta(noisy_tensor, tta_count=8):
    """Run model with test-time augmentation."""
    predictions = []
    for i in range(tta_count):
        transformed = apply_tta_transform(noisy_tensor, i)
        pred = model(transformed)
        pred = invert_tta_transform(pred, i)
        predictions.append(pred)
    avg = torch.stack(predictions, dim=0).mean(dim=0)
    return torch.clamp(avg, 0.0, 1.0)


# ============================================================
# API ENDPOINT
# ============================================================


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_number = data.get("image_number", "").strip()

    # Normalize to filename
    if image_number.endswith(".npy"):
        image_number = image_number[:-4]

    if not image_number.isdigit():
        return jsonify({"error": "Please enter a valid numeric image number."}), 400

    filename = f"{int(image_number):06d}.npy"

    noisy_path = None
    gt_path = None
    source = "input"

    # Search only in the CLI-specified input directory
    direct_path = os.path.join(CLI_INPUT_DIR, filename)
    sub_noisy_path = os.path.join(CLI_INPUT_DIR, "NoisyLR", filename)
    train_sub_path = os.path.join(CLI_INPUT_DIR, "train", "train", "NoisyLR", filename)

    if os.path.exists(direct_path):
        noisy_path = direct_path
    elif os.path.exists(sub_noisy_path):
        noisy_path = sub_noisy_path
    elif os.path.exists(train_sub_path):
        noisy_path = train_sub_path
    else:
        return jsonify({
            "error": f"Image '{filename}' not found in input directory: {CLI_INPUT_DIR}"
        }), 404

    # Check for GT inside the input directory
    direct_gt = os.path.join(CLI_INPUT_DIR, "GT", filename)
    train_gt = os.path.join(CLI_INPUT_DIR, "train", "train", "GT", filename)
    if os.path.exists(direct_gt):
        gt_path = direct_gt
    elif os.path.exists(train_gt):
        gt_path = train_gt

    # Load noisy input
    noisy_np = np.load(noisy_path).astype(np.float32)
    noisy_tensor = (
        torch.from_numpy(noisy_np)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # Run inference
    with torch.no_grad():
        sr_standard = torch.clamp(model(noisy_tensor), 0.0, 1.0)
        sr_tta = predict_tta(noisy_tensor, tta_count=8)

    sr_standard_np = sr_standard.cpu().numpy().squeeze()
    sr_tta_np = sr_tta.cpu().numpy().squeeze()

    # Save output to specified output directory
    out_save_path = None
    if CLI_OUTPUT_DIR:
        os.makedirs(CLI_OUTPUT_DIR, exist_ok=True)
        out_save_path = os.path.join(CLI_OUTPUT_DIR, filename)
        save_arr = np.nan_to_num(sr_tta_np, nan=0.0, posinf=1.0, neginf=0.0)
        save_arr = np.clip(save_arr, 0.0, 1.0).astype(np.float32)
        np.save(out_save_path, save_arr)

    response = {
        "filename": filename,
        "source": source,
        "saved_to": out_save_path,
        "noisy_image": numpy_to_base64_png(noisy_np),
        "noisy_shape": list(noisy_np.shape),
        "sr_standard_image": numpy_to_base64_png(sr_standard_np),
        "sr_tta_image": numpy_to_base64_png(sr_tta_np),
        "sr_shape": list(sr_tta_np.shape),
    }

    # If GT exists, compute metrics
    if gt_path and os.path.exists(gt_path):
        gt_np = np.load(gt_path).astype(np.float32)
        gt_tensor = (
            torch.from_numpy(gt_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            psnr_std = calculate_psnr(sr_standard, gt_tensor)
            ssim_std = ssim(sr_standard, gt_tensor).item()
            lpips_std = calculate_lpips(sr_standard, gt_tensor)
            psnr_tta = calculate_psnr(sr_tta, gt_tensor)
            ssim_tta = ssim(sr_tta, gt_tensor).item()
            lpips_tta = calculate_lpips(sr_tta, gt_tensor)

        error_np = np.abs(sr_tta_np - gt_np)

        response["gt_image"] = numpy_to_base64_png(gt_np)
        response["gt_shape"] = list(gt_np.shape)
        response["error_image"] = numpy_to_base64_png(
            error_np / max(error_np.max(), 1e-6)
        )
        response["metrics"] = {
            "standard": {
                "psnr": round(psnr_std, 4),
                "ssim": round(ssim_std, 6),
                "lpips": round(lpips_std, 6),
            },
            "tta_8x": {
                "psnr": round(psnr_tta, 4),
                "ssim": round(ssim_tta, 6),
                "lpips": round(lpips_tta, 6),
            },
        }

    return jsonify(response)


# ============================================================
# HTML PAGE
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SemiSR — Semiconductor Image Super-Resolution Viewer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-card: #1a2035;
            --bg-card-hover: #1f2847;
            --bg-input: #0d1220;
            --border-subtle: rgba(99, 135, 255, 0.12);
            --border-focus: rgba(99, 135, 255, 0.5);
            --accent-primary: #6387ff;
            --accent-secondary: #38bdf8;
            --accent-gradient: linear-gradient(135deg, #6387ff 0%, #38bdf8 50%, #a78bfa 100%);
            --accent-glow: rgba(99, 135, 255, 0.25);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --success: #34d399;
            --warning: #fbbf24;
            --error: #f87171;
            --radius: 16px;
            --radius-sm: 10px;
            --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.4), 0 0 40px rgba(99, 135, 255, 0.06);
            --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
            --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99, 135, 255, 0.07) 0%, transparent 60%),
                radial-gradient(ellipse 60% 50% at 80% 90%, rgba(56, 189, 248, 0.05) 0%, transparent 60%),
                radial-gradient(ellipse 50% 40% at 50% 50%, rgba(167, 139, 250, 0.04) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }

        .app-container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 24px 60px;
        }

        /* ---- HEADER ---- */
        .header {
            text-align: center;
            margin-bottom: 40px;
            animation: fadeInDown 0.6s ease-out;
        }

        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            border-radius: 100px;
            background: rgba(99, 135, 255, 0.08);
            border: 1px solid var(--border-subtle);
            font-size: 12px;
            font-weight: 500;
            color: var(--accent-primary);
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }

        .header-badge .dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }

        .header h1 {
            font-size: 42px;
            font-weight: 800;
            letter-spacing: -1px;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 16px;
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ---- INPUT SECTION ---- */
        .input-section {
            max-width: 640px;
            margin: 0 auto 48px;
            animation: fadeInUp 0.6s ease-out 0.15s both;
        }

        .input-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius);
            padding: 28px 32px;
            box-shadow: var(--shadow-card);
            transition: var(--transition);
        }

        .input-card:hover {
            border-color: var(--border-focus);
        }

        .input-row {
            display: flex;
            gap: 12px;
            align-items: stretch;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        .input-wrapper input {
            width: 100%;
            padding: 16px 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 500;
            background: var(--bg-input);
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            outline: none;
            transition: var(--transition);
        }

        .input-wrapper input::placeholder {
            color: var(--text-muted);
            font-family: 'Inter', sans-serif;
            font-weight: 400;
        }

        .input-wrapper input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .predict-btn {
            padding: 16px 32px;
            background: var(--accent-gradient);
            border: none;
            border-radius: var(--radius-sm);
            color: white;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            white-space: nowrap;
            position: relative;
            overflow: hidden;
        }

        .predict-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 30px rgba(99, 135, 255, 0.35);
        }

        .predict-btn:active { transform: translateY(0); }

        .predict-btn.loading {
            pointer-events: none;
            opacity: 0.85;
        }

        .predict-btn.loading::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
            animation: shimmer 1.2s infinite;
        }

        .input-hints {
            display: flex;
            gap: 16px;
            margin-top: 14px;
            flex-wrap: wrap;
        }

        .hint {
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .hint-tag {
            padding: 2px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 500;
        }

        .hint-tag.test { background: rgba(56, 189, 248, 0.12); color: var(--accent-secondary); }
        .hint-tag.train { background: rgba(52, 211, 153, 0.12); color: var(--success); }

        /* ---- ERROR / STATUS ---- */
        .status-message {
            text-align: center;
            padding: 16px;
            border-radius: var(--radius-sm);
            margin-bottom: 32px;
            font-size: 14px;
            font-weight: 500;
            animation: fadeIn 0.3s ease-out;
        }

        .status-message.error {
            background: rgba(248, 113, 113, 0.08);
            border: 1px solid rgba(248, 113, 113, 0.2);
            color: var(--error);
        }

        /* ---- RESULTS ---- */
        .results-container {
            animation: fadeInUp 0.5s ease-out;
        }

        .source-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 20px;
        }

        .source-badge.test { background: rgba(56, 189, 248, 0.1); color: var(--accent-secondary); border: 1px solid rgba(56, 189, 248, 0.2); }
        .source-badge.train { background: rgba(52, 211, 153, 0.1); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.2); }
        .source-badge.validation { background: rgba(251, 191, 36, 0.1); color: var(--warning); border: 1px solid rgba(251, 191, 36, 0.2); }

        /* ---- METRICS PANEL ---- */
        .metrics-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 20px 24px;
            text-align: center;
            transition: var(--transition);
        }

        .metric-card:hover {
            border-color: var(--border-focus);
            transform: translateY(-2px);
            box-shadow: var(--shadow-card);
        }

        .metric-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .metric-unit {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .metric-mode {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-top: 8px;
        }

        .metric-mode.std { background: rgba(99, 135, 255, 0.1); color: var(--accent-primary); }
        .metric-mode.tta { background: rgba(167, 139, 250, 0.1); color: #a78bfa; }

        /* ---- IMAGE GRID ---- */
        .image-grid {
            display: grid;
            gap: 20px;
            margin-top: 8px;
        }

        .image-grid.cols-2 { grid-template-columns: 1fr 1fr; }
        .image-grid.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
        .image-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }

        .image-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius);
            overflow: hidden;
            transition: var(--transition);
        }

        .image-panel:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }

        .image-panel.panel-input {
            border: 1.5px solid rgba(148, 163, 184, 0.25);
        }
        .image-panel.panel-restored {
            border: 2px solid rgba(56, 189, 248, 0.6);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.12);
        }
        .image-panel.panel-tta {
            border: 2px solid rgba(167, 139, 250, 0.7);
            box-shadow: 0 0 24px rgba(167, 139, 250, 0.18);
        }
        .image-panel.panel-gt {
            border: 1.5px solid rgba(52, 211, 153, 0.4);
        }
        .image-panel.panel-error {
            border: 1.5px solid rgba(251, 191, 36, 0.4);
        }

        .image-panel-header {
            padding: 14px 18px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(15, 23, 42, 0.6);
        }

        .image-panel-title {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .image-panel-subtitle {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .panel-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .panel-badge.badge-input { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }
        .panel-badge.badge-restored { background: rgba(56, 189, 248, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
        .panel-badge.badge-tta { background: rgba(167, 139, 250, 0.25); color: #c084fc; border: 1px solid rgba(167, 139, 250, 0.5); }
        .panel-badge.badge-gt { background: rgba(52, 211, 153, 0.18); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
        .panel-badge.badge-error { background: rgba(251, 191, 36, 0.18); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }

        .image-panel-footer {
            padding: 8px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            background: rgba(10, 14, 26, 0.8);
            border-top: 1px solid var(--border-subtle);
            text-align: right;
        }

        .image-panel-body {
            padding: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #080c16;
        }

        .image-panel-body img {
            width: 100%;
            height: auto;
            display: block;
            image-rendering: pixelated;
            border-radius: 4px;
        }

        /* ---- COMPARISON SLIDER ---- */
        .comparison-section {
            margin-top: 32px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-title .icon {
            width: 28px; height: 28px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }

        /* ---- FOOTER ---- */
        .footer {
            text-align: center;
            margin-top: 60px;
            padding-top: 24px;
            border-top: 1px solid var(--border-subtle);
            color: var(--text-muted);
            font-size: 13px;
        }

        .footer span {
            color: var(--accent-primary);
            font-weight: 600;
        }

        /* ---- LOADING SKELETON ---- */
        .skeleton {
            background: linear-gradient(90deg, var(--bg-card) 25%, var(--bg-card-hover) 50%, var(--bg-card) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: var(--radius);
            min-height: 300px;
        }

        /* ---- ANIMATIONS ---- */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* ---- RESPONSIVE ---- */
        @media (max-width: 900px) {
            .image-grid.cols-3 { grid-template-columns: 1fr; }
            .image-grid.cols-4 { grid-template-columns: 1fr 1fr; }
            .header h1 { font-size: 30px; }
            .metrics-panel { grid-template-columns: 1fr 1fr; }
        }

        @media (max-width: 600px) {
            .input-row { flex-direction: column; }
            .image-grid.cols-2 { grid-template-columns: 1fr; }
            .image-grid.cols-4 { grid-template-columns: 1fr; }
            .metrics-panel { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="app-container">

        <!-- HEADER -->
        <header class="header">
            <div class="header-badge">
                <span class="dot"></span>
                Model Active &middot; RTX 4050
            </div>
            <h1>SemiSR Viewer</h1>
            <p>Semiconductor image super-resolution powered by <strong>RestorationNetV2</strong> with 20 residual blocks &amp; 8&times; test-time augmentation</p>
        </header>

        <!-- INPUT -->
        <section class="input-section">
            <div class="input-card">
                <div class="input-row">
                    <div class="input-wrapper">
                        <input
                            type="text"
                            id="imageNumberInput"
                            placeholder="Enter image number (e.g. 42, 001093, 000250)"
                            autocomplete="off"
                        />
                    </div>
                    <button class="predict-btn" id="predictBtn" onclick="runPrediction()">
                        Restore &amp; View
                    </button>
                </div>
                <div class="input-hints">
                    <span class="hint"><span class="hint-tag test">Input Dir</span> {{ input_dir }}</span>
                </div>
            </div>
        </section>

        <!-- STATUS -->
        <div id="statusArea"></div>

        <!-- RESULTS -->
        <div id="resultsArea"></div>

        <!-- FOOTER -->
        <footer class="footer">
            <strong>best_model_v2_blocks20.pth</strong> &middot; RestorationNetV2 &middot; 1.66M params &middot; Epoch 120<br>
            Built with <span>&hearts;</span> Flask + PyTorch
        </footer>
    </div>

    <script>
        const input = document.getElementById('imageNumberInput');
        const btn = document.getElementById('predictBtn');
        const statusArea = document.getElementById('statusArea');
        const resultsArea = document.getElementById('resultsArea');

        // Allow Enter key
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') runPrediction();
        });

        async function runPrediction() {
            const imageNumber = input.value.trim();
            if (!imageNumber) {
                showStatus('Please enter an image number.', 'error');
                return;
            }

            btn.classList.add('loading');
            btn.textContent = 'Processing…';
            statusArea.innerHTML = '';
            resultsArea.innerHTML = `
                <div class="image-grid cols-2">
                    <div class="skeleton"></div>
                    <div class="skeleton"></div>
                </div>
            `;

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_number: imageNumber })
                });

                const data = await response.json();

                if (!response.ok) {
                    showStatus(data.error || 'Something went wrong.', 'error');
                    resultsArea.innerHTML = '';
                    return;
                }

                renderResults(data);

            } catch (err) {
                showStatus('Network error: ' + err.message, 'error');
                resultsArea.innerHTML = '';
            } finally {
                btn.classList.remove('loading');
                btn.textContent = 'Restore & View';
            }
        }

        function showStatus(message, type) {
            statusArea.innerHTML = `<div class="status-message ${type}">${message}</div>`;
        }

        function renderResults(data) {
            statusArea.innerHTML = '';
            let html = '<div class="results-container">';

            // Source badge
            const sourceLabel = {
                input: '📂 Input Directory Image',
                test: '🔬 Test Dataset Image',
                train: '🏋️ Training Set Image',
                validation: '✅ Validation Set Image',
                train_data: '📁 Training Data Image'
            };
            const sourceClass = data.source === 'validation' ? 'validation' :
                                data.source === 'test' ? 'test' : 'train';

            html += `<div class="source-badge ${sourceClass}">${sourceLabel[data.source] || data.source} &mdash; ${data.filename}</div>`;

            // Metrics (if GT available)
            if (data.metrics) {
                html += `
                <div class="metrics-panel">
                    <div class="metric-card">
                        <div class="metric-label">PSNR (Standard)</div>
                        <div class="metric-value">${data.metrics.standard.psnr.toFixed(2)}</div>
                        <div class="metric-unit">dB</div>
                        <div class="metric-mode std">1× Forward</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">PSNR (8× TTA)</div>
                        <div class="metric-value">${data.metrics.tta_8x.psnr.toFixed(2)}</div>
                        <div class="metric-unit">dB</div>
                        <div class="metric-mode tta">8× TTA</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">SSIM (Standard)</div>
                        <div class="metric-value">${data.metrics.standard.ssim.toFixed(4)}</div>
                        <div class="metric-unit">Structural Similarity</div>
                        <div class="metric-mode std">1× Forward</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">SSIM (8× TTA)</div>
                        <div class="metric-value">${data.metrics.tta_8x.ssim.toFixed(4)}</div>
                        <div class="metric-unit">Structural Similarity</div>
                        <div class="metric-mode tta">8× TTA</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">LPIPS (Standard)</div>
                        <div class="metric-value">${data.metrics.standard.lpips.toFixed(4)}</div>
                        <div class="metric-unit">Perceptual Distance ↓</div>
                        <div class="metric-mode std">1× Forward</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">LPIPS (8× TTA)</div>
                        <div class="metric-value">${data.metrics.tta_8x.lpips.toFixed(4)}</div>
                        <div class="metric-unit">Perceptual Distance ↓</div>
                        <div class="metric-mode tta">8× TTA</div>
                    </div>
                </div>`;
            }

            // Images
            if (data.gt_image) {
                // With GT: show 4 panels
                html += `
                <div class="section-title">
                    <span class="icon" style="background:rgba(99,135,255,0.12);">🖼️</span>
                    Image Restoration Comparison
                </div>
                <div class="image-grid cols-4">
                    ${imagePanel('📥 Original Input', 'Noisy Low-Res', 'INPUT', 'badge-input', 'panel-input', data.noisy_shape.join('×'), data.noisy_image)}
                    ${imagePanel('✨ Restored Image', 'Standard Model (1×)', 'RESTORED', 'badge-restored', 'panel-restored', data.sr_shape.join('×'), data.sr_standard_image)}
                    ${imagePanel('🚀 Restored Image', 'Enhanced (8× TTA)', 'BEST RESTORED', 'badge-tta', 'panel-tta', data.sr_shape.join('×'), data.sr_tta_image)}
                    ${imagePanel('🎯 Ground Truth', 'Clean Reference', 'TARGET GT', 'badge-gt', 'panel-gt', data.gt_shape.join('×'), data.gt_image)}
                </div>
                <div class="image-grid cols-2" style="margin-top:20px;">
                    ${imagePanel('📊 Error Heatmap', 'Absolute Difference (Restored vs GT)', 'DIFFERENCE', 'badge-error', 'panel-error', data.sr_shape.join('×'), data.error_image)}
                </div>`;
            } else {
                // Test set: no GT
                html += `
                <div class="section-title">
                    <span class="icon" style="background:rgba(56,189,248,0.12);">🔬</span>
                    Image Restoration Result
                </div>
                <div class="image-grid cols-3">
                    ${imagePanel('📥 Original Input', 'Noisy Low-Res', 'INPUT', 'badge-input', 'panel-input', data.noisy_shape.join('×'), data.noisy_image)}
                    ${imagePanel('✨ Restored Image', 'Standard Model (1×)', 'RESTORED', 'badge-restored', 'panel-restored', data.sr_shape.join('×'), data.sr_standard_image)}
                    ${imagePanel('🚀 Restored Image', 'Enhanced (8× TTA)', 'BEST RESTORED', 'badge-tta', 'panel-tta', data.sr_shape.join('×'), data.sr_tta_image)}
                </div>`;
            }

            html += '</div>';
            resultsArea.innerHTML = html;
        }

        function imagePanel(title, subtitle, badgeText, badgeClass, panelClass, resolution, base64Data) {
            return `
            <div class="image-panel ${panelClass}">
                <div class="image-panel-header">
                    <div>
                        <div class="image-panel-title">${title}</div>
                        <div class="image-panel-subtitle">${subtitle}</div>
                    </div>
                    <span class="panel-badge ${badgeClass}">${badgeText}</span>
                </div>
                <div class="image-panel-body">
                    <img src="data:image/png;base64,${base64Data}" alt="${title}" />
                </div>
                <div class="image-panel-footer">
                    <span>${resolution}</span>
                </div>
            </div>`;
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, input_dir=os.path.abspath(CLI_INPUT_DIR))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"SemiSR Web Viewer — Starting at http://localhost:{cli_args.port}")
    if CLI_INPUT_DIR:
        print(f"Input Directory  : {os.path.abspath(CLI_INPUT_DIR)}")
    if CLI_OUTPUT_DIR:
        print(f"Output Directory : {os.path.abspath(CLI_OUTPUT_DIR)}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=cli_args.port, debug=False)
