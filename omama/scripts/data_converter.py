from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from tqdm import tqdm

import os
import glob
import argparse
import numpy as np
import skimage.io as mh
import skimage.transform as skt


def split_image(img):
    """
    Splits the image into two halves.

    Args:
    img : numpy.ndarray
        The image to split.

    Returns:
    left, right : tuple of numpy.ndarray
        The left and right halves of the image.
    """
    left = img[0:512, 0:512]
    right = img[0:512, 512:]
    return left, right


def resize_image(side, normalize):
    """
    Resizes and normalizes an image.

    Args:
    side : numpy.ndarray
        The image to resize.
    normalize : bool
        Whether to normalize the image.

    Returns:
    side_resized : numpy.ndarray
        The resized and possibly normalized image.
    """
    side_resized = skt.resize(side, (512, 512), preserve_range=True)
    side_resized = side_resized.astype(np.float32)
    if normalize:
        side_resized /= 255.0
    return side_resized


def resize_and_cast(side, target_dtype):
    """
    Resizes an image and changes its dtype.

    Args:
    side : numpy.ndarray
        The image to resize.
    target_dtype : type
        The dtype to cast the image to.

    Returns:
    side_resized : numpy.ndarray
        The resized and casted image.
    """
    side_resized = skt.resize(side, (512, 512), preserve_range=True)
    side_resized = side_resized.astype(target_dtype)
    return side_resized


def process_image(img, normalize):
    """
    Processes an image by converting it to grayscale, splitting it in half,
    resizing and normalizing each half, and returning them as a numpy array.

    Args:
    img : numpy.ndarray
        The image to process.
    normalize : bool
        Whether to normalize the image.

    Returns:
    numpy.ndarray
        An array containing the processed halves of the image.
    """
    # Check if the image has more than one channel
    if len(img.shape) > 2 and img.shape[2] > 1:
        # If image has alpha channel, convert it to RGB
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        # Convert image to grayscale
        img = rgb2gray(img)
    left, right = split_image(img)
    images = []
    left_resized = resize_image(left, normalize)
    if np.prod(left_resized.shape) > 0:
        images.append(left_resized)
    if right.size != 0:  # Add this line
        right_resized = resize_image(right, normalize)
        if np.prod(right_resized.shape) > 0:
            images.append(right_resized)
    return np.array(images)


def process_mask(img):
    """
    Processes a mask by converting it to grayscale, splitting it in half,
    and resizing each half.

    Args:
    img : numpy.ndarray
        The mask to process.

    Returns:
    numpy.ndarray
        An array containing the processed halves of the mask.
    """
    # Check if the image has more than one channel
    if len(img.shape) > 2 and img.shape[2] > 1:
        # If image has alpha channel, convert it to RGB
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        # Convert image to grayscale
        img = rgb2gray(img)
    left, right = split_image(img)
    masks = []
    left_resized = resize_and_cast(left, bool)
    if np.prod(left_resized.shape) > 0:
        masks.append(left_resized)
    if right.size != 0:  # Add this line
        right_resized = resize_and_cast(right, bool)
        if np.prod(right_resized.shape) > 0:
            masks.append(right_resized)
    return np.array(masks)


def load_files(datafolder, file_type, mask_identifier, normalize):
    """
    Loads all image or mask files from a folder, processes them,
    and returns them as a numpy array.

    Args:
    datafolder : str
        The path to the folder containing the files.
    file_type : str
        The type of file to load ('image' or 'mask').
    mask_identifier : str
        The identifier in the filename to recognize mask files.
    normalize : bool
        Whether to normalize the images.

    Returns:
    numpy.ndarray
        An array containing the processed files.
    """
    all_files = sorted(glob.glob(os.path.join(datafolder, "*.*")))
    # If mask_identifier is not provided, don't use it for filtering
    if mask_identifier is None:
        image_files = all_files if file_type == "image" else []
        mask_files = all_files if file_type == "mask" else []
    else:
        image_files = (
            [file for file in all_files if mask_identifier not in file]
            if file_type == "image"
            else []
        )
        mask_files = (
            [file for file in all_files if mask_identifier in file]
            if file_type == "mask"
            else []
        )

    if file_type == "image":
        images = []
        for a in tqdm(image_files):
            img = mh.imread(a)
            images.extend(process_image(img, normalize))
        return np.array(images)

    elif file_type == "mask":
        masks = []
        for a in tqdm(mask_files):
            img = mh.imread(a)
            masks.extend(process_mask(img))
        return np.array(masks)


def process_images_and_masks(
    image_folder,
    mask_folder,
    mask_identifier,
    normalize,
    output_dir,
    image_filename,
    mask_filename,
    lower_threshold_percentage=0.05,
    upper_threshold_percentage=0.95,
):
    """
    Loads, processes, and saves image and mask data.

    Args:
    image_folder : str
        The path to the folder containing the images.
    mask_folder : str
        The path to the folder containing the masks.
    mask_identifier : str
        The identifier in the filename to recognize mask files.
    normalize : bool
        Whether to normalize the images.
    output_dir : str
        The path to the directory to save the processed data in.
    image_filename : str
        The filename for the processed image array.
    mask_filename : str
        The filename for the processed mask array.
    lower_threshold_percentage : float
        The lower threshold for the mask.
    upper_threshold_percentage : float
        The upper threshold for the mask.
    """
    images = load_files(image_folder, "image", mask_identifier, normalize)
    images = np.expand_dims(images, axis=-1)
    print("Image dtype:", images.dtype)
    print("Image shape:", images.shape)

    if mask_folder is None:
        print("skip mask loading")
    else:
        masks = load_files(mask_folder, "mask", mask_identifier, normalize)
        total_pixels = np.prod(masks[0].shape)
        lower_threshold_value = lower_threshold_percentage * total_pixels
        upper_threshold_value = upper_threshold_percentage * total_pixels
        valid_indices = [
            i
            for i in range(len(masks))
            if lower_threshold_value < np.sum(masks[i]) < upper_threshold_value
        ]
        masks = masks[valid_indices]
    # Keep only images and masks with some segmentation

    # if images.shape[0] != masks.shape[0]:
    #     raise ValueError("Number of images and masks do not match")

    # masks = np.expand_dims(masks, axis=-1)
    # print("Mask dtype:", masks.dtype)
    # print("Mask shape:", masks.shape)

    np.save(os.path.join(output_dir, image_filename), images)
    # np.save(os.path.join(output_dir, mask_filename), masks)


def main():
    """
    Defines the command line arguments, parses them, and calls
    process_images_and_masks with the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load and process image and mask data."
    )
    parser.add_argument(
        "-i", "--image_folder", type=str, help="Root path to the images."
    )
    parser.add_argument(
        "-m",
        "--mask_folder",
        type=str,
        default=None,
        help="Root path to the masks. If not specified, it is assumed that masks are in the image folder.",
    )
    parser.add_argument(
        "-mi",
        "--mask_identifier",
        type=str,
        default=None,
        help="Identifier in the filename to recognize mask files.",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Whether to normalize the images."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for the processed arrays.",
    )
    parser.add_argument(
        "-if",
        "--image_filename",
        type=str,
        help="Filename for the processed image array.",
    )
    parser.add_argument(
        "-mf",
        "--mask_filename",
        type=str,
        help="Filename for the processed mask array.",
    )
    parser.add_argument(
        "--lower_threshold_percentage",
        type=float,
        default=0.01,
        help="Lower threshold percentage for valid masks.",
    )
    parser.add_argument(
        "--upper_threshold_percentage",
        type=float,
        default=0.99,
        help="Upper threshold percentage for valid masks.",
    )

    args = parser.parse_args()

    process_images_and_masks(
        args.image_folder,
        args.mask_folder,
        args.mask_identifier,
        args.normalize,
        args.output_dir,
        args.image_filename,
        args.mask_filename,
        args.lower_threshold_percentage,
        args.upper_threshold_percentage,
    )


if __name__ == "__main__":
    main()
