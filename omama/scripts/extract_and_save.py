import os
import numpy as np
import pydicom
from PIL import Image
import argparse
from multiprocessing import Pool, Manager
from numba import jit
import gc
from tqdm import tqdm


@jit(nopython=True)
def process_2d_image(pixel_array):
    """Process 2D image pixel array."""
    return pixel_array


@jit(nopython=True)
def process_3d_image(pixel_array):
    """Process 3D image pixel array."""
    return pixel_array


def determine_image_type(pixel_array):
    array_shape = pixel_array.shape
    if len(array_shape) == 2:
        return "2D"
    elif len(array_shape) == 3:
        return "3D"
    else:
        return None


def worker(dicom_path, _output_folder, _prefix_to_remove=""):
    # Removing the prefix from the filename
    basename = os.path.basename(dicom_path).replace(_prefix_to_remove, "")
    output_filename = f"{basename}.npz"
    output_path = os.path.join(_output_folder, output_filename)

    if os.path.exists(output_path):
        print(f"{output_filename} already exists. Skipping.")
        return

    ds = pydicom.dcmread(dicom_path)

    if not hasattr(ds, "PixelData"):
        print(f"{dicom_path} does not have PixelData. Skipping.")
        return

    if not (hasattr(ds, "Rows") and hasattr(ds, "Columns")):
        print(f"{dicom_path} does not have Rows and Columns attributes. Skipping.")
        return

    pixel_array = ds.pixel_array
    image_type = determine_image_type(pixel_array)

    if image_type == "2D":
        processed_image = process_2d_image(pixel_array)
    elif image_type == "3D":
        processed_image = process_3d_image(pixel_array)
    else:
        print(f"Unknown image type for {dicom_path}. Skipping.")
        return

    np.savez_compressed(output_path, data=processed_image)

    del pixel_array, processed_image, ds
    gc.collect()


def worker_with_counter_and_lock(
    dicom_path, _output_folder, counter, lock, _prefix_to_remove
):
    with lock:
        counter.value += 1
        if counter.value % 1000 == 0:
            tqdm.write(f"Processed: {counter.value}", end="\r")
    worker(dicom_path, _output_folder, _prefix_to_remove)


def extract_and_save_pixels(
    _input_txt_file, _output_folder, _num_processes, _max_tasks, _prefix_to_remove
):
    with open(_input_txt_file, "r") as f:
        dicom_paths = f.read().splitlines()

    manager = Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    with Pool(processes=_num_processes, maxtasksperchild=_max_tasks) as pool:
        pool.starmap(
            worker_with_counter_and_lock,
            [
                (dicom_path, _output_folder, counter, lock, _prefix_to_remove)
                for dicom_path in dicom_paths
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and save DICOM image pixel data to compressed numpy arrays."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input TXT file containing DICOM paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output folder for saving numpy arrays.",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to all available CPUs.",
    )
    parser.add_argument(
        "-m",
        "--maxtasks",
        type=int,
        default=None,
        help="Maximum number of tasks per child process. Defaults to infinite.",
    )
    parser.add_argument(
        "-r",
        "--removeprefix",
        type=str,
        default="",
        help="Filename prefix to remove before saving the numpy arrays.",
    )

    args = parser.parse_args()

    input_txt_file = args.input
    output_folder = args.output
    num_processes = args.processes
    max_tasks = args.maxtasks
    prefix_to_remove = args.removeprefix

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Processing {len(open(args.input, 'r').readlines())} DICOM files...")
    extract_and_save_pixels(
        input_txt_file, output_folder, num_processes, max_tasks, prefix_to_remove
    )
    print("Done.")
