import os
import torch
import pandas as pd
import numpy as np
from torchvision import models
from torch import nn
from tqdm import tqdm
from preprocess import WasilPreprocessor
from utils import get_device, seed_everything, ensure_dir
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle


def extract_features(metadata_path, strips_dir, output_dir):
    seed_everything()
    device = get_device()
    ensure_dir(output_dir)

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove classification head (keep up to Global Average Pooling)
    model = nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    model.eval()

    preprocessor = WasilPreprocessor()

    features = []
    labels = []
    page_ids = []

    print("Extracting CNN features...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(strips_dir, row["filename"])
            try:
                tensor_img = (
                    preprocessor.preprocess_image(img_path).unsqueeze(0).to(device)
                )
                feat = model(tensor_img).cpu().numpy().flatten()
                features.append(feat)
                labels.append(row["book_id"])
                page_ids.append(row["page_id"])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    features = np.array(features)

    # Apply PCA and Scaling
    print(f"Original feature shape: {features.shape}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA to 256 components
    n_components = min(256, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    print(f"Reduced feature shape: {features_pca.shape}")

    # Save results
    np.save(os.path.join(output_dir, "features_pca.npy"), features_pca)

    # Save labels and page_ids
    df_labels = pd.DataFrame(
        {"filename": df["filename"], "book_id": labels, "page_id": page_ids}
    )
    df_labels.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    # Save PCA and Scaler for future use
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"Features and labels saved to {output_dir}")


if __name__ == "__main__":
    extract_features(
        "wasil_project/dataset/metadata.csv",
        "wasil_project/dataset/strips_3mm",
        "wasil_project/outputs/features",
    )
