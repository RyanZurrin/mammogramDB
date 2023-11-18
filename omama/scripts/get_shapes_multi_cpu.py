import os
import numpy as np
import argparse
import concurrent.futures
from collections import Counter


def count_shapes(npz_file):
    try:
        data = np.load(npz_file)
        pixel_array = data["data"]
        shape = pixel_array.shape
        return shape
    except Exception as e:
        return None


def save_shape_counts(shape_counts, output_file):
    with open(output_file, "w") as file:
        for shape, count in shape_counts.items():
            file.write(f"Shape {shape}: Count {count}\n")


def main(input_dir, num_threads, output_file):
    npz_files = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith(".npz")
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        shapes = list(executor.map(count_shapes, npz_files))

    # Filter out None values (failed reads)
    shapes = [shape for shape in shapes if shape is not None]

    # Count the unique shapes
    shape_counts = Counter(shapes)

    # Print the shape counts
    for shape, count in shape_counts.items():
        print(f"Shape {shape}: Count {count}")

    if output_file:
        save_shape_counts(shape_counts, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count shapes of pixel arrays in NPZ files."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing NPZ files.",
    )
    parser.add_argument(
        "-t", "--threads", type=int, required=True, help="Number of CPU threads to use."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output file for saving shape counts.",
    )
    args = parser.parse_args()

    input_dir = args.input
    num_threads = args.threads
    output_file = args.output

    main(input_dir, num_threads, output_file)
