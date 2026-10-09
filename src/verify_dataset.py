"""
verify_dataset.py
-----------------
Dataset inspection and validation script for Garbage Classification CNN project.

Performs dataset integrity checks:
1. Verifies class directories for the 6 expected categories.
2. Counts valid image files (.jpg, .jpeg, .png, .bmp, .webp).
3. Inspects image dimensions (min, max, average, and most common size).
4. Identifies corrupted/unreadable images using Pillow.
5. Reports missing classes and outputs a clear validation summary.
"""

import argparse
from collections import Counter
from pathlib import Path
from PIL import Image

EXPECTED_CLASSES = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash'
]

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_class_directories(dataset_root: Path):
    """
    Locates directories corresponding to expected classes.
    First checks direct subdirectories, then searches recursively if needed.
    """
    class_dirs = {}
    for cls in EXPECTED_CLASSES:
        direct_dir = dataset_root / cls
        nested_dir = dataset_root / "dataset-resized" / cls
        if direct_dir.is_dir():
            class_dirs[cls] = [direct_dir]
        elif nested_dir.is_dir():
            class_dirs[cls] = [nested_dir]
        else:
            # Recursive search for folder matching class name (case-insensitive)
            found = [
                d for d in dataset_root.rglob("*")
                if d.is_dir() and d.name.lower() == cls.lower() and "__MACOSX" not in d.parts
            ]
            class_dirs[cls] = found
    return class_dirs


def verify_dataset(dataset_path_str: str = "dataset"):
    """
    Main function to inspect and validate the garbage classification dataset.
    """
    dataset_root = Path(dataset_path_str).resolve()
    
    print("========================================")
    print("DATASET VERIFICATION REPORT")
    print("========================================")
    print(f"Target directory: {dataset_root}\n")

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"ERROR: Dataset directory '{dataset_root}' does not exist or is not a directory.")
        print("\nClass distribution:")
        max_label_len = max(len(c) for c in EXPECTED_CLASSES)
        for cls in EXPECTED_CLASSES:
            print(f"  - {cls:<{max_label_len}} : 0")
        print("\nTotal images: 0")
        print("Corrupted images: 0")
        print(f"Missing classes: {', '.join(EXPECTED_CLASSES)}")
        print("\n========================================")
        print("DATASET VALIDATION FAILED")
        print("========================================")
        return False

    class_dirs = find_class_directories(dataset_root)
    
    class_counts = {}
    missing_classes = []
    all_dimensions = []
    corrupted_images = []
    total_valid_images = 0

    for cls in EXPECTED_CLASSES:
        dirs = class_dirs.get(cls, [])
        if not dirs:
            missing_classes.append(cls)
            class_counts[cls] = 0
            continue
        
        image_files = []
        for d in dirs:
            for filepath in d.rglob("*"):
                if filepath.is_file() and not filepath.name.startswith('.') and filepath.suffix.lower() in VALID_EXTENSIONS and "__MACOSX" not in filepath.parts:
                    image_files.append(filepath)

        valid_class_count = 0
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                with Image.open(img_path) as img:
                    all_dimensions.append(img.size)
                    valid_class_count += 1
            except Exception:
                corrupted_images.append(img_path)

        class_counts[cls] = valid_class_count
        if valid_class_count == 0:
            missing_classes.append(cls)

        total_valid_images += valid_class_count

    print("Class distribution:")
    max_label_len = max(len(c) for c in EXPECTED_CLASSES)
    for cls in EXPECTED_CLASSES:
        print(f"  - {cls:<{max_label_len}} : {class_counts[cls]} images")

    print(f"\nTotal valid images: {total_valid_images}")

    print("\nImage dimension statistics:")
    if all_dimensions:
        widths = [dim[0] for dim in all_dimensions]
        heights = [dim[1] for dim in all_dimensions]
        
        min_dim = (min(widths), min(heights))
        max_dim = (max(widths), max(heights))
        avg_dim = (round(sum(widths) / len(widths), 1), round(sum(heights) / len(heights), 1))
        most_common_size, most_common_count = Counter(all_dimensions).most_common(1)[0]
        
        print(f"  - Minimum dimensions : {min_dim[0]}x{min_dim[1]}")
        print(f"  - Maximum dimensions : {max_dim[0]}x{max_dim[1]}")
        print(f"  - Average dimensions : {avg_dim[0]}x{avg_dim[1]}")
        print(f"  - Most common size   : {most_common_size[0]}x{most_common_size[1]} ({most_common_count} images)")
    else:
        print("  - Dimensions: N/A")

    print(f"\nCorrupted images: {len(corrupted_images)}")
    if corrupted_images:
        print("Corrupted file list:")
        for corrupt_path in corrupted_images[:10]:
            print(f"  - {corrupt_path}")
        if len(corrupted_images) > 10:
            print(f"  ... and {len(corrupted_images) - 10} more.")

    if missing_classes:
        unique_missing = list(dict.fromkeys(missing_classes))
        print(f"\nMissing classes: {', '.join(unique_missing)}")
    else:
        print("\nMissing classes: None")

    print("\n========================================")
    print("DATASET VALIDATION COMPLETE")
    print("========================================")
    return total_valid_images > 0 and len(missing_classes) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and inspect Garbage Classification dataset.")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="dataset",
        help="Path to dataset directory (default: 'dataset')"
    )
    args = parser.parse_args()
    verify_dataset(args.dataset_path)
