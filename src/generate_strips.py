import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def generate_strips(
    input_dir,
    output_dir,
    strip_width_mm=30,
    page_width_mm=140,
    num_strips=30,
    margin_ratio=0.0,
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    metadata = []

    page_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if not page_files:
        print(f"No images found in {input_dir}")
        return

    for page_file in tqdm(page_files, desc="Generating strips"):
        page_id = os.path.splitext(page_file)[0]
        book_id = "mushaf"

        img_path = os.path.join(input_dir, page_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        h, w = img.shape[:2]

        strip_width_px = max(1, int((strip_width_mm / page_width_mm) * w))

        # Ignore outer white margins slightly
        x_min = int(w * margin_ratio)
        x_max = int(w * (1 - margin_ratio)) - strip_width_px

        if x_max <= x_min:
            x_min = 0
            x_max = max(0, w - strip_width_px)

        if x_max <= x_min:
            print(f"Skipping {page_file}: strip width too large for image width.")
            continue

        # Uniform coverage across the page + small random jitter
        base_positions = np.linspace(x_min, x_max, num=num_strips).astype(int)

        step = max(1, (x_max - x_min) // max(1, num_strips))
        jitter_limit = max(1, step // 3)

        x_starts = []
        for x in base_positions:
            jitter = rng.integers(-jitter_limit, jitter_limit + 1)
            x_jittered = int(np.clip(x + jitter, x_min, x_max))
            x_starts.append(x_jittered)

        # Remove duplicates while preserving order
        x_starts = list(dict.fromkeys(x_starts))

        for i, x_start in enumerate(x_starts):
            x_end = x_start + strip_width_px
            strip = img[:, x_start:x_end]

            strip_filename = f"{page_id}_strip_{i:03d}_{strip_width_mm}mm.png"
            strip_path = os.path.join(output_dir, strip_filename)
            cv2.imwrite(strip_path, strip)

            metadata.append(
                {
                    "filename": strip_filename,
                    "book_id": book_id,
                    "page_id": page_id,
                    "strip_width_mm": strip_width_mm,
                    "strip_width_px": strip_width_px,
                    "x_start": x_start,
                    "x_end": x_end,
                    "image_width_px": w,
                    "image_height_px": h,
                }
            )

    df = pd.DataFrame(metadata)
    metadata_path = os.path.join(os.path.dirname(output_dir), "metadata.csv")
    df.to_csv(metadata_path, index=False)

    print(f"Generated {len(metadata)} strips and saved to {output_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="dataset/pages", help="Path to full page images"
    )
    parser.add_argument(
        "--output", default="dataset/strips_30mm", help="Path to save strips"
    )
    parser.add_argument(
        "--strip_width_mm", type=float, default=30, help="Effective strip width in mm"
    )
    parser.add_argument(
        "--page_width_mm", type=float, default=140, help="Physical page width in mm"
    )
    parser.add_argument(
        "--num_strips", type=int, default=30, help="Number of strips per page"
    )
    parser.add_argument(
        "--margin_ratio", type=float, default=0.05, help="Ignore page margins ratio"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_strips(
        input_dir=args.input,
        output_dir=args.output,
        strip_width_mm=args.strip_width_mm,
        page_width_mm=args.page_width_mm,
        num_strips=args.num_strips,
        margin_ratio=args.margin_ratio,
        seed=args.seed,
    )
