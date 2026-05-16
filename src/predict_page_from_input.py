import os
import re
import argparse
import pickle
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from transformers import BeitImageProcessor, BeitModel
import matplotlib.pyplot as plt


def find_page_file(pages_dir, page_number):
    """
    Finds a page image like:
    mushaf_page_001.jpg
    mushaf_page_001.png
    """
    page_str = f"{page_number:03d}"

    valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"]

    candidates = []
    for filename in os.listdir(pages_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_exts:
            continue

        if page_str in name:
            candidates.append(filename)

    if not candidates:
        raise FileNotFoundError(
            f"No page image found for page {page_number} in {pages_dir}"
        )

    candidates = sorted(candidates)
    return os.path.join(pages_dir, candidates[0])


def generate_single_strip(
    page_path,
    output_dir,
    page_number,
    strip_width_mm=30,
    page_width_mm=140,
    x_mode="center",
    x_start=None,
):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(page_path)
    if img is None:
        raise ValueError(f"Could not read image: {page_path}")

    h, w = img.shape[:2]

    strip_width_px = max(1, int((strip_width_mm / page_width_mm) * w))

    if strip_width_px >= w:
        raise ValueError("Strip width is larger than image width.")

    max_x = w - strip_width_px

    if x_start is not None:
        x_start = int(np.clip(x_start, 0, max_x))
    elif x_mode == "center":
        x_start = max_x // 2
    elif x_mode == "left":
        x_start = 0
    elif x_mode == "right":
        x_start = max_x
    elif x_mode == "random":
        x_start = np.random.randint(0, max_x + 1)
    else:
        raise ValueError("x_mode must be one of: center, left, right, random")

    x_end = x_start + strip_width_px
    strip = img[:, x_start:x_end]

    strip_filename = (
        f"test_mushaf_page_{page_number:03d}_{strip_width_mm}mm_x{x_start}.png"
    )
    strip_path = os.path.join(output_dir, strip_filename)

    cv2.imwrite(strip_path, strip)

    return {
        "strip_path": strip_path,
        "page_path": page_path,
        "actual_page": f"mushaf_page_{page_number:03d}",
        "page_number": page_number,
        "x_start": x_start,
        "x_end": x_end,
        "strip_width_px": strip_width_px,
        "image_width_px": w,
        "image_height_px": h,
    }


def load_dit_feature_extractor(model_name="microsoft/dit-base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = BeitImageProcessor.from_pretrained(model_name)
    model = BeitModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return processor, model, device


def extract_single_dit_feature(strip_path, processor, model, device):
    image = Image.open(strip_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    feature = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return feature.reshape(1, -1)


def load_scaler_and_pca(features_dir):
    scaler_path_options = [
        os.path.join(features_dir, "scaler_dit.pkl"),
        os.path.join(features_dir, "scaler.pkl"),
    ]

    pca_path_options = [
        os.path.join(features_dir, "pca_dit.pkl"),
        os.path.join(features_dir, "pca.pkl"),
    ]

    scaler_path = next((p for p in scaler_path_options if os.path.exists(p)), None)
    pca_path = next((p for p in pca_path_options if os.path.exists(p)), None)

    if scaler_path is None:
        raise FileNotFoundError(f"No scaler file found in {features_dir}")

    if pca_path is None:
        raise FileNotFoundError(f"No PCA file found in {features_dir}")

    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    try:
        pca = joblib.load(pca_path)
    except Exception:
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

    return scaler, pca


def transform_feature(feature, scaler, pca):
    feature_scaled = scaler.transform(feature)
    feature_pca = pca.transform(feature_scaled)
    return feature_pca


def load_models(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]

    if not model_files:
        raise FileNotFoundError(f"No *_model.pkl files found in {models_dir}")

    models = {}

    for model_file in sorted(model_files):
        model_name = model_file.replace("_model.pkl", "")
        model_path = os.path.join(models_dir, model_file)

        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)

    return models


def get_top_k_predictions(model, feature_pca, k=5):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(feature_pca)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(feature_pca)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        scores = scores[0]

        # softmax-like conversion for ranking
        scores = scores - np.max(scores)
        scores = np.exp(scores)
        scores = scores / np.sum(scores)
    else:
        pred = model.predict(feature_pca)[0]
        return [(pred, None)]

    classes = model.classes_
    top_indices = np.argsort(scores)[::-1][:k]

    return [(classes[i], float(scores[i])) for i in top_indices]


def extract_page_number_from_label(label):
    label = str(label)
    digits = re.findall(r"\d+", label)
    if not digits:
        return None
    return int(digits[-1])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--page_number", type=int, required=True)
    parser.add_argument("--pages_dir", required=True)
    parser.add_argument("--test_strips_dir", default="outputs/test_strips")

    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--models_dir", required=True)

    parser.add_argument("--strip_width_mm", type=float, default=30)
    parser.add_argument("--page_width_mm", type=float, default=140)

    parser.add_argument(
        "--x_mode",
        default="center",
        choices=["center", "left", "right", "random"],
        help="Where to crop the strip from the page",
    )

    parser.add_argument(
        "--x_start", type=int, default=None, help="Optional exact x_start in pixels"
    )

    parser.add_argument("--model_name", default="microsoft/dit-base")
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    page_path = find_page_file(args.pages_dir, args.page_number)

    strip_info = generate_single_strip(
        page_path=page_path,
        output_dir=args.test_strips_dir,
        page_number=args.page_number,
        strip_width_mm=args.strip_width_mm,
        page_width_mm=args.page_width_mm,
        x_mode=args.x_mode,
        x_start=args.x_start,
    )

    processor, dit_model, device = load_dit_feature_extractor(args.model_name)

    raw_feature = extract_single_dit_feature(
        strip_path=strip_info["strip_path"],
        processor=processor,
        model=dit_model,
        device=device,
    )

    scaler, pca = load_scaler_and_pca(args.features_dir)
    feature_pca = transform_feature(raw_feature, scaler, pca)

    models = load_models(args.models_dir)
    strip_img = Image.open(strip_info["strip_path"]).convert("RGB")

    print("\n==============================")
    print("Wasil Single Page Test")
    print("==============================")
    print(f"Page image:       {strip_info['page_path']}")
    print(f"Generated strip:  {strip_info['strip_path']}")
    print(f"Correct page:     {strip_info['actual_page']}")
    print(f"x_start/x_end:    {strip_info['x_start']} / {strip_info['x_end']}")
    print(f"strip width px:   {strip_info['strip_width_px']}")
    print("==============================\n")

    plt.figure(figsize=(4, 10))
    plt.imshow(strip_img)
    plt.axis("off")
    plt.title(f"Generated Strip\n" f"Correct: {strip_info['actual_page']}")
    plt.show()

    rows = []

    for model_name, model in models.items():
        prediction = model.predict(feature_pca)[0]
        top_k = get_top_k_predictions(model, feature_pca, k=args.top_k)

        actual_num = args.page_number
        pred_num = extract_page_number_from_label(prediction)

        abs_error = None
        if pred_num is not None:
            abs_error = abs(pred_num - actual_num)

        print(f"Model: {model_name}")
        print(f"Predicted page: {prediction}")
        print(f"Absolute page error: {abs_error}")

        print(f"Top-{args.top_k}:")
        for rank, (label, score) in enumerate(top_k, start=1):
            if score is None:
                print(f"  {rank}. {label}")
            else:
                print(f"  {rank}. {label} | score={score:.4f}")

        print()

        rows.append(
            {
                "model": model_name,
                "correct_page": strip_info["actual_page"],
                "predicted_page": prediction,
                "absolute_page_error": abs_error,
                "strip_path": strip_info["strip_path"],
                "x_start": strip_info["x_start"],
                "x_end": strip_info["x_end"],
            }
        )

    results_df = pd.DataFrame(rows)

    os.makedirs(args.test_strips_dir, exist_ok=True)
    result_csv = os.path.join(
        args.test_strips_dir, f"prediction_page_{args.page_number:03d}.csv"
    )
    results_df.to_csv(result_csv, index=False)

    print(f"Prediction summary saved to: {result_csv}")


if __name__ == "__main__":
    main()
