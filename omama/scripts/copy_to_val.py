import os
import shutil
import argparse
from pathlib import Path


def copy_images(src_train_dir, dest_train_dir, dest_val_dir, num_images_to_copy):
    src_train_images_dir = Path(src_train_dir) / "images"
    src_train_masks_dir = Path(src_train_dir) / "masks"

    dest_train_images_dir = Path(dest_train_dir) / "images"
    dest_val_images_dir = Path(dest_val_dir) / "images"
    dest_val_masks_dir = Path(dest_val_dir) / "masks"

    # Create the val directories if they don't exist
    dest_val_images_dir.mkdir(parents=True, exist_ok=True)
    dest_val_masks_dir.mkdir(parents=True, exist_ok=True)

    # Get the set of image filenames in the destination train directory
    dest_train_images = set(os.listdir(dest_train_images_dir))

    # Get all image filenames from the source train directory
    src_train_images = os.listdir(src_train_images_dir)

    # Initialize a counter for the number of images to copy
    copied_images_count = 0

    # Copy images and masks to the val directory
    for image_name in src_train_images:
        if copied_images_count >= num_images_to_copy:
            break

        # Skip if the image already exists in the destination train directory
        if image_name in dest_train_images:
            continue

        # Construct the source paths
        image_src = src_train_images_dir / image_name
        mask_src = src_train_masks_dir / image_name.replace(".npz", "_mask.npz")

        # Construct the destination paths
        image_dst = dest_val_images_dir / image_name
        mask_dst = dest_val_masks_dir / image_name.replace(".npz", "_mask.npz")

        # Copy the image and mask
        shutil.copy(image_src, image_dst)
        shutil.copy(mask_src, mask_dst)

        copied_images_count += 1

    print(f"Done copying {copied_images_count} images and masks to the validation set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy images and masks to validation set."
    )
    parser.add_argument(
        "src_train_dir", help="The directory of the source training images and masks."
    )
    parser.add_argument(
        "dest_train_dir", help="The directory of the destination training images."
    )
    parser.add_argument(
        "dest_val_dir",
        help="The directory where the validation images and masks will be copied.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=2000,
        help="Number of images to copy to the validation set.",
    )

    args = parser.parse_args()

    copy_images(
        args.src_train_dir, args.dest_train_dir, args.dest_val_dir, args.num_images
    )
