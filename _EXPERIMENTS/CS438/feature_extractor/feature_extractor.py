import os
import pydicom
from PIL import Image
import numpy as np
import pandas as pd
import argparse
from multiprocessing import Pool
from tqdm.auto import tqdm
import json

FEATURES_TO_EXTRACT = [
    "StudyInstanceUID",
    "ImageLaterality",
    "SOPInstanceUID",
    "PatientAge",
    "Manufacturer",
    "ManufacturerModelName",
    "DistanceSourceToDetector",
    "DistanceSourceToPatient",
    "ExposureTime",
    "XRayTubeCurrent",
    "Exposure",
    "ExposureInuAs",
    "KVP",
    "BodyPartThickness",
    "CompressionForce",
    "PositionerPrimaryAngle",
    "ViewPosition",
    "DetectorTemperature",
    "DetectorType",
    "FieldOfViewOrigin",
    "Rows",
    "Columns",
    "PixelSpacing",
    "BreastImplantPresent",
    "WindowCenter",
    "WindowWidth",
    "HalfValueLayer",
]

CACHE_PATH = "/home/ryan.zurrin001/binlink/feature_cache.pkl"
PREDICTIONS_PATH = (
    "/home/ryan.zurrin001/Projects/omama/omama/deep_sight/predictions_cache.json"
)
LABEL_PATHS = [
    "/home/ryan.zurrin001/Projects/omama/_EXPERIMENTS/CS438/labels/dh_dcm_ast_labels.csv",
    "/home/ryan.zurrin001/Projects/omama/_EXPERIMENTS/CS438/labels/dh_dh0new_labels.csv",
    "/home/ryan.zurrin001/Projects/omama/_EXPERIMENTS/CS438/labels/dh_dh2_labels.csv",
]


# Function to check if running in a SLURM environment
def is_slurm_job():
    return "SLURM_JOB_ID" in os.environ


# Wrap tqdm to only enable progress bar if not in a SLURM job
# Wrap tqdm to only enable progress bar if not in a SLURM job
def tqdm_maybe(iterable, *args, **kwargs):
    if is_slurm_job():
        return iterable  # Return the iterable directly if SLURM job detected
    else:
        return tqdm(iterable, *args, **kwargs)  # Otherwise, return tqdm iterator


def load_predictions(json_path):
    with open(json_path, "r") as file:
        predictions = json.load(file)
    return predictions


def load_labels(csv_files):
    labels = {}
    for file in csv_files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            key = (row["StudyInstanceUID"], row["CancerLaterality"])
            labels[key] = row["Label"]
    return labels


def resize_coords(coord, original_size, target_size):
    """Resize a single set of bounding box coordinates."""
    if not coord or len(coord) != 4:
        print(f"Invalid coordinates format: {coord}")
        return []

    # Calculate scaling factors
    scale_y = target_size[0] / original_size[0]
    scale_x = target_size[1] / original_size[1]

    # Apply scaling to coordinates
    resized_coord = (
        int(coord[0] * scale_x),
        int(coord[1] * scale_y),
        int(coord[2] * scale_x),
        int(coord[3] * scale_y),
    )

    return resized_coord


def compute_histogram(image, bins=255):
    """Compute histogram of an image."""
    histogram, _ = np.histogram(image, bins=bins, range=(0, 255))
    histogram_sum = np.sum(histogram)
    if histogram_sum > 0:
        return histogram / histogram_sum
    else:
        return np.zeros_like(histogram)


def read_dicom(file_path, features):
    """Read a DICOM file and extract specified metadata and image data."""
    ds = pydicom.dcmread(file_path)

    # Extract specified metadata
    metadata = [getattr(ds, feature, None) for feature in features]

    # Extract pixel data
    pixel_array = ds.pixel_array

    return metadata, pixel_array


def resize_image(pixel_array, target_size):
    """Resize the image data without distortion."""
    # Check if the pixel_array is in a format that PIL can handle, convert if necessary
    if pixel_array.dtype != np.uint8:
        # Normalize and convert to 8-bit
        pixel_array = (
            (pixel_array - pixel_array.min()) / (pixel_array.ptp() / 255.0)
        ).astype(np.uint8)

    # Handle different number of channels
    if len(pixel_array.shape) == 2:
        # Single channel (grayscale)
        image = Image.fromarray(pixel_array, mode="L")
    elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
        # Three channels (RGB)
        image = Image.fromarray(pixel_array, mode="RGB")
    else:
        # Unsupported format
        raise ValueError("Unsupported image format")

    # Resize image
    image = image.resize(target_size, Image.ANTIALIAS)
    return np.array(image)


def process_dicom_file(args):
    file_path, features, target_size, labels, predictions, store_image = args
    if os.path.isfile(file_path):
        metadata, pixel_array = read_dicom(file_path, features)
        original_shape = pixel_array.shape
        resized_pixels = resize_image(pixel_array, target_size)
        study_instance_uid = metadata[
            0
        ]  # Assuming StudyInstanceUID is the first feature
        image_laterality = metadata[1]  # Assuming ImageLaterality is the second feature
        sop_instance_uid = metadata[2]  # Assuming SOPInstanceUID is the first feature

        # Determine the label
        label = labels.get(
            (study_instance_uid, image_laterality),
            labels.get((study_instance_uid, "None"), "Unknown"),
        )

        # Determine the prediction
        prediction = predictions.get(sop_instance_uid, {"coords": [], "score": 0})

        # Resize the coordinates and save as new resized coordinates
        resized_coords = resize_coords(
            prediction["coords"], original_shape, target_size
        )

        # Calculate histogram for the image
        grayscale_image = resized_pixels
        if len(grayscale_image.shape) == 3:
            grayscale_image = np.mean(grayscale_image, axis=2).astype(np.uint8)
        else:
            grayscale_image = grayscale_image.astype(np.uint8)

        histogram = compute_histogram(np.array(grayscale_image))

        return {
            "path": file_path,
            "metadata": metadata,
            "label": label,
            "original_shape": original_shape,
            "shape": target_size,
            "coords": prediction["coords"],
            "resized_coords": resized_coords,
            "score": prediction["score"],
            "histogram": histogram,
            "image": resized_pixels if store_image else None,
        }
    else:
        print(f"File not found: {file_path}")
        return None


def process_dicom_files(
    file_list,
    features,
    target_size,
    labels,
    predictions,
    num_cores,
    cache_path=None,
    store_images=False,
):
    file_paths = [line.strip() for line in open(file_list, "r")]
    args = [
        (file_path, features, target_size, labels, predictions, store_images)
        for file_path in file_paths
    ]
    # Check if cache exists
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_pickle(cache_path)

    # Process DICOM files in parallel using Pool
    pool = Pool(num_cores)
    results = list(
        tqdm_maybe(
            pool.imap_unordered(process_dicom_file, args),
            total=len(args),
            desc="Processing DICOM Files",
        )
    )
    pool.close()
    pool.join()

    results = [result for result in results if result is not None]

    # Create a DataFrame from the processed results
    # First, create a dictionary with separate keys for each feature
    results_dict = {
        "path": [],
        "label": [],
        **{feature: [] for feature in features},
        "original_shape": [],
        "shape": [],
        "coords": [],
        "resized_coords": [],
        "score": [],
        "histogram": [],
        "image": [],
    }

    # Create a DataFrame from the processed results making  each feature a column
    columns = (
        ["path", "label"]
        + features
        + [
            "original_shape",
            "shape",
            "coords",
            "resized_coords",
            "score",
            "histogram",
        ]
    )
    if store_images:
        columns.append("image")
    df = pd.DataFrame(columns=columns)

    # Populate the dictionary with the results
    for result in results:
        if result is not None:
            row_data = {
                "path": result["path"],
                "label": result["label"],
                **{
                    feature: result["metadata"][i] for i, feature in enumerate(features)
                },
                "original_shape": result["original_shape"],
                "shape": result["shape"],
                "coords": tuple(result["coords"]),
                "resized_coords": tuple(result["resized_coords"]),
                "score": result["score"],
                "histogram": tuple(result["histogram"]),
            }
            if store_images:
                row_data["image"] = result["image"]

            # Append the row to the DataFrame
            df = df.append(row_data, ignore_index=True)

    df.drop_duplicates(subset=["path"], inplace=True)

    # If a cache path is provided, save the DataFrame
    if cache_path:
        df.to_pickle(cache_path)
        print(f"Feature extraction complete and saved to '{cache_path}'")

    return df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process DICOM files and return a structured dictionary."
    )
    parser.add_argument(
        "file_list", type=str, help="Text file with paths to DICOM files"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help="List of DICOM features to extract",
        default=FEATURES_TO_EXTRACT,
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        help="Target size for image resizing",
        default=(256, 256),
    )
    parser.add_argument(
        "--csv_files",
        nargs="+",
        help="List of CSV files with labels",
        default=[
            "/home/ryan.zurrin001/Projects/omama/_EXPERIMENTS/CS438/labels/dh_dcm_ast_labels.csv",
            "/home/ryan.zurrin001/Projects/omama/_EXPERIMENTS/CS438/labels/dh_dh0new_labels.csv",
            "/home/ryan.zurrin001/Projects/omama/_EXPERIMENTS/CS438/labels/dh_dh2_labels.csv",
        ],
    )
    parser.add_argument(
        "--cores", type=int, help="Number of CPU cores to use", default=1
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Path to save the DataFrame as a pickle file",
        default="/home/ryan.zurrin001/binlink/cs438_feature_cache.pkl",
    )
    parser.add_argument(
        "--preds_path",
        type=str,
        help="Path to the predictions JSON file",
        default="/home/ryan.zurrin001/Projects/omama/omama/deep_sight/predictions_cache.json",
    )
    parser.add_argument(
        "--store_images",
        action="store_true",
        help="Store the images in the DataFrame",
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    labels = load_labels(args.csv_files)
    predictions = load_predictions(args.preds_path)
    dicom_features_df = process_dicom_files(
        args.file_list,
        args.features,
        tuple(args.size),
        labels,
        predictions,
        args.cores,
        args.cache_path,
    )

    print(dicom_features_df.head())

    print("Feature extraction complete.")


# For Jupyter notebook usage, define a separate function
def process_dicoms_in_notebook(
    file_list,
    size=(256, 256),
    csv_files=None,
    cores=4,
    features=None,
    preds_path=PREDICTIONS_PATH,
    cache_path=CACHE_PATH,
    store_images=True,
    force=False,
):
    if force:
        if os.path.exists(cache_path):
            os.remove(cache_path)
    if features is None:
        features = FEATURES_TO_EXTRACT
    if csv_files is None:
        csv_files = LABEL_PATHS
    labels = load_labels(csv_files)
    predictions = load_predictions(preds_path)
    return process_dicom_files(
        file_list, features, size, labels, predictions, cores, cache_path, store_images
    )


if __name__ == "__main__":
    main()
