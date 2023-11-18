import os
import shutil
import random
import argparse
from tqdm import tqdm


def split_data_into_train_test(image_folder, mask_folder, dest_folder, train_ratio=0.8):
    # List all the files in the image_folder
    all_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".npz")])
    random.shuffle(all_files)

    # Calculate split indices
    num_files = len(all_files)
    num_train = int(num_files * train_ratio)
    num_test = num_files - num_train

    # Create destination directories
    train_img_folder = os.path.join(dest_folder, "train/images")
    test_img_folder = os.path.join(dest_folder, "test/images")
    train_mask_folder = os.path.join(dest_folder, "train/masks")
    test_mask_folder = os.path.join(dest_folder, "test/masks")

    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(test_img_folder, exist_ok=True)
    os.makedirs(train_mask_folder, exist_ok=True)
    os.makedirs(test_mask_folder, exist_ok=True)

    # Move files
    for i, file in tqdm(enumerate(all_files)):
        img_src_path = os.path.join(image_folder, file)
        mask_src_path = os.path.join(mask_folder, file.replace(".npz", "_mask.npz"))

        if i < num_train:
            img_dest_path = os.path.join(train_img_folder, file)
            mask_dest_path = os.path.join(
                train_mask_folder, file.replace(".npz", "_mask.npz")
            )
        else:
            img_dest_path = os.path.join(test_img_folder, file)
            mask_dest_path = os.path.join(
                test_mask_folder, file.replace(".npz", "_mask.npz")
            )

        shutil.copy(img_src_path, img_dest_path)
        shutil.copy(mask_src_path, mask_dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into training and testing sets."
    )
    parser.add_argument(
        "-i", "--image_folder",
        type=str,
        help="Path to the folder containing the image files.",
    )
    parser.add_argument(
        "-m", "--mask_folder", type=str, help="Path to the folder containing the mask files."
    )
    parser.add_argument(
        "-d", "--dest_folder",
        type=str,
        help="Path to the destination folder where train/test folders will be created.",
    )
    parser.add_argument(
        "-r", "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data to overall data.",
    )
    args = parser.parse_args()

    split_data_into_train_test(
        args.image_folder, args.mask_folder, args.dest_folder, args.train_ratio
    )
