import os
import argparse
from pathlib import Path


def rename_mushaf_pages(pages_dir, prefix="mushaf_page", start_index=1, dry_run=True):
    pages_path = Path(pages_dir)

    if not pages_path.exists():
        raise FileNotFoundError(f"Directory not found: {pages_dir}")

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

    files = sorted(
        [
            f
            for f in pages_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not files:
        print("No image files found.")
        return

    print(f"Found {len(files)} image files.")

    # Step 1: rename to temporary names to avoid overwrite conflicts
    temp_files = []

    for i, file_path in enumerate(files):
        temp_name = f"__temp_rename_{i:04d}{file_path.suffix.lower()}"
        temp_path = pages_path / temp_name

        if dry_run:
            print(f"[DRY RUN] {file_path.name} -> {temp_name}")
        else:
            file_path.rename(temp_path)

        temp_files.append(
            (temp_path if not dry_run else file_path, file_path.suffix.lower())
        )

    # Step 2: rename temporary names to final names
    for i, (file_path, suffix) in enumerate(temp_files, start=start_index):
        new_name = f"{prefix}_{i:03d}{suffix}"
        new_path = pages_path / new_name

        if dry_run:
            print(f"[DRY RUN] {file_path.name} -> {new_name}")
        else:
            file_path.rename(new_path)
            print(f"{file_path.name} -> {new_name}")

    print("Done." if not dry_run else "Dry run complete. No files were renamed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pages_dir", required=True, help="Directory containing Mushaf page images"
    )

    parser.add_argument(
        "--prefix", default="mushaf_page", help="Output filename prefix"
    )

    parser.add_argument(
        "--start_index", type=int, default=1, help="Starting page number"
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only dry-run is performed.",
    )

    args = parser.parse_args()

    rename_mushaf_pages(
        pages_dir=args.pages_dir,
        prefix=args.prefix,
        start_index=args.start_index,
        dry_run=not args.apply,
    )
