import shutil
import random
from pathlib import Path

# Paths
original_dataset = Path("dataset")
output_dir = Path("dataset_split")

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create output folders
for split in ["train", "val", "test"]:
    (output_dir / split).mkdir(parents=True, exist_ok=True)


def copy_images(img_paths, dest_folder):
    dest_folder.mkdir(parents=True, exist_ok=True)
    for img in img_paths:
        # Extract magnification folder name if it exists
        try:
            mag_name = img.parent.name  # e.g., magnification_40
        except IndexError:
            mag_name = "unknown"

        # Create a new filename: magName_originalFileName
        new_name = f"{mag_name}_{img.name}"

        shutil.copy(img, dest_folder / new_name)


# Iterate through disease folders inside benign/SOB and malignant/SOB
for tissue_type in ["benign", "malignant"]:
    sob_path = original_dataset / tissue_type / "SOB"
    if not sob_path.exists():
        print(f"⚠ Missing expected folder: {sob_path}")
        continue

    # Each disease folder inside SOB is a class
    for disease_folder in sob_path.iterdir():
        if disease_folder.is_dir():
            class_name = disease_folder.name

            # Gather all images inside this disease folder (including subfolders)
            all_images = list(disease_folder.rglob("*.*"))

            if not all_images:
                print(f"⚠ No images found in: {disease_folder}")
                continue

            random.shuffle(all_images)

            total = len(all_images)
            train_count = int(total * train_ratio)
            val_count = int(total * val_ratio)

            train_imgs = all_images[:train_count]
            val_imgs = all_images[train_count : train_count + val_count]
            test_imgs = all_images[train_count + val_count :]

            copy_images(train_imgs, output_dir / "train" / class_name)
            copy_images(val_imgs, output_dir / "val" / class_name)
            copy_images(test_imgs, output_dir / "test" / class_name)

print("✅ Dataset split complete!")
