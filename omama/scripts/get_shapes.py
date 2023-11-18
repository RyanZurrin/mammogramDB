import os
import numpy as np
import concurrent.futures
import argparse
from collections import Counter


def count_shapes(npz_file):
    try:
        data = np.load(npz_file)
        pixel_array = data["data"]
        shape = pixel_array.shape
        return shape
    except Exception as e:
        print(f"Error processing {npz_file}: {e}")
        return None


def main():
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
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Number of worker threads (default: 8).",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to the output file for saving results."
    )
    args = parser.parse_args()

    input_dir = args.input
    num_threads = args.threads
    output_file = args.output

    # List all NPZ files in the directory
    npz_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".npz")
    ]

    # Use concurrent.futures.ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each NPZ file
        future_to_file = {
            executor.submit(count_shapes, npz_file): npz_file for npz_file in npz_files
        }

        # Wait for all tasks to complete and collect results
        shapes = []
        for future in concurrent.futures.as_completed(future_to_file):
            npz_file = future_to_file[future]
            try:
                shape = future.result()
                if shape is not None:
                    shapes.append(shape)
            except Exception as e:
                print(f"Error processing {npz_file}: {e}")

    # Count the occurrences of each shape using Counter
    shape_counter = Counter(shapes)

    # Now you have a dictionary with shape frequencies
    for shape, count in shape_counter.items():
        print(f"Shape {shape}: Count {count}")

    # Save the results to the output file if specified
    if output_file:
        with open(output_file, "w") as f:
            for shape, count in shape_counter.items():
                f.write(f"Shape {shape}: Count {count}\n")


if __name__ == "__main__":
    main()
