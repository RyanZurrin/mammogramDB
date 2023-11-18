import os
import numpy as np
import json
from skimage.transform import resize
import argparse
from tqdm import tqdm


from scipy import stats


def resize_and_pad_image(image, target_size=(512, 512)):
    # First, resize the image while maintaining the aspect ratio
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = resize(image, (new_h, new_w), order=1, preserve_range=True)

    # Calculate mean values for each half of the image
    upper_half = resized_image[: new_h // 2, :]
    lower_half = resized_image[new_h // 2 :, :]
    left_half = resized_image[:, : new_w // 2]
    right_half = resized_image[:, new_w // 2 :]

    upper_half_mean = np.mean(upper_half)
    lower_half_mean = np.mean(lower_half)
    left_half_mean = np.mean(left_half)
    right_half_mean = np.mean(right_half)

    # Determine the padding values based on darker side
    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h

    # Determine padding color from the darker side
    if left_half_mean < right_half_mean:
        pad_x = (delta_w, 0)
        pad_color_x = stats.mode(left_half, axis=None)[0][0]
    else:
        pad_x = (0, delta_w)
        pad_color_x = stats.mode(right_half, axis=None)[0][0]

    if upper_half_mean < lower_half_mean:
        pad_y = (delta_h, 0)
        pad_color_y = stats.mode(upper_half, axis=None)[0][0]
    else:
        pad_y = (0, delta_h)
        pad_color_y = stats.mode(lower_half, axis=None)[0][0]

    # Pad the resized image with the mode pixel value of the darker side
    padded_image = np.pad(
        resized_image,
        [(pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])],
        mode="constant",
        constant_values=((pad_color_y, pad_color_y), (pad_color_x, pad_color_x)),
    )

    # Return the paddings too for further processing
    return padded_image.astype(np.uint16), (pad_y, pad_x)


def resize_and_save_images(input_dir, output_dir, target_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    for npz_file in tqdm(npz_files, desc="Processing NPZ files"):
        data = np.load(os.path.join(input_dir, npz_file))
        pixel_array = data["data"]

        h, w = pixel_array.shape
        scale = min(target_size[0] / h, target_size[1] / w)

        resized_image, (pad_y, pad_x) = resize_and_pad_image(pixel_array, target_size)

        json_filename = f"{os.path.splitext(npz_file)[0]}.json"
        json_path = os.path.join(input_dir, json_filename)

        with open(json_path, "r") as json_file_handle:
            metadata = json.load(json_file_handle)

        coords = metadata["coords"]

        # Adjust the bounding box coordinates based on scaling
        coords[0] = int(coords[0] * scale)
        coords[2] = int(coords[2] * scale)
        coords[1] = int(coords[1] * scale)
        coords[3] = int(coords[3] * scale)

        # Adjust the bounding box coordinates based on padding
        coords[0] += pad_x[0]
        coords[2] += pad_x[0]
        coords[1] += pad_y[0]
        coords[3] += pad_y[0]

        output_npz_file = os.path.join(output_dir, npz_file)
        np.savez_compressed(output_npz_file, data=resized_image)

        metadata["coords"] = coords

        output_json_file = os.path.join(output_dir, json_filename)
        with open(output_json_file, "w") as json_file_handle:
            json.dump(metadata, json_file_handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and pad images using skimage.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input directory containing NPZ and JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output directory for resized images and updated JSON files.",
    )
    parser.add_argument(
        "-w", "--width", type=int, default=512, help="Target width for resized images."
    )
    parser.add_argument(
        "-ht",
        "--height",
        type=int,
        default=512,
        help="Target height for resized images.",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    target_size = (args.width, args.height)

    resize_and_save_images(input_dir, output_dir, target_size)
