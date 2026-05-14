import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def generate_strips(input_dir, output_dir, strip_width_mm=30, page_width_mm=140):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata = []
    page_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    )

    if not page_files:
        print(f"No images found in {input_dir}")
        return

    for page_file in tqdm(page_files, desc="Generating strips"):
        # Assuming filename format: page_001.png or similar
        # Extract page_id from filename
        page_id = os.path.splitext(page_file)[0]
        book_id = "mushaf"  # Default for now

        img_path = os.path.join(input_dir, page_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        strip_width_px = int((strip_width_mm / page_width_mm) * w)

        # Sliding window with some overlap or random positions
        # Let's use a fixed number of strips per page for the demo
        num_strips = 20
        # Calculate possible x_start positions
        possible_x = range(0, w - strip_width_px)

        # Use a deterministic seed for reproducibility in this experiment
        np.random.seed(42)
        x_starts = np.random.choice(
            possible_x, size=min(num_strips, len(possible_x)), replace=False
        )

        for i, x_start in enumerate(x_starts):
            x_end = x_start + strip_width_px
            strip = img[:, x_start:x_end]

            strip_filename = f"{book_id}_{page_id}_strip_{i:03d}_{strip_width_mm}mm.png"
            strip_path = os.path.join(output_dir, strip_filename)
            cv2.imwrite(strip_path, strip)

            metadata.append(
                {
                    "filename": strip_filename,
                    "book_id": book_id,
                    "page_id": page_id,
                    "strip_width_mm": strip_width_mm,
                    "x_start": x_start,
                    "x_end": x_end,
                }
            )

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(os.path.dirname(output_dir), "metadata.csv"), index=False)
    print(f"Generated {len(metadata)} strips and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="dataset/pages", help="Path to full page images"
    )
    parser.add_argument(
        "--output", default="dataset/strips_3mm", help="Path to save strips"
    )
    args = parser.parse_args()

    # Adjust paths relative to project root if needed
    generate_strips(args.input, args.output)
