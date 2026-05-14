import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BeitImageProcessor, BeitModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


def extract_features_dit(
    metadata_path,
    strips_dir,
    output_dir,
    model_name="microsoft/dit-base",
    pca_components=256,
):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(metadata_path)

    processor = BeitImageProcessor.from_pretrained(model_name)
    model = BeitModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    features = []
    labels = []
    filenames = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting DiT features"):
        img_path = os.path.join(strips_dir, row["filename"])

        if not os.path.exists(img_path):
            print(f"Missing file: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # CLS token embedding: shape = [hidden_dim]
            feature = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

            features.append(feature)
            labels.append(row["page_id"])
            filenames.append(row["filename"])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    features = np.array(features)

    if len(features) == 0:
        raise ValueError(
            "No features were extracted. Check metadata filenames and strips_dir."
        )

    print("Raw DiT feature shape:", features.shape)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_samples, n_features = features_scaled.shape
    actual_pca_components = min(pca_components, n_samples, n_features)

    pca = PCA(n_components=actual_pca_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)

    print("PCA feature shape:", features_pca.shape)

    np.save(os.path.join(output_dir, "features_pca.npy"), features_pca)

    labels_df = pd.DataFrame({"filename": filenames, "page_id": labels})
    labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    joblib.dump(scaler, os.path.join(output_dir, "scaler_dit.pkl"))
    joblib.dump(pca, os.path.join(output_dir, "pca_dit.pkl"))

    print(f"Saved DiT features to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="dataset/metadata.csv")
    parser.add_argument("--strips_dir", default="dataset/strips_30mm")
    parser.add_argument("--output_dir", default="outputs/features_dit")
    parser.add_argument("--model_name", default="microsoft/dit-base")
    parser.add_argument("--pca_components", type=int, default=256)
    args = parser.parse_args()

    extract_features_dit(
        metadata_path=args.metadata,
        strips_dir=args.strips_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pca_components=args.pca_components,
    )
