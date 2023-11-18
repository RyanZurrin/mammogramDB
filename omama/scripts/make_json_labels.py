import os
import json
import argparse
from tqdm import tqdm


def main(npz_directory, json_file, prefix):
    # Load JSON data
    with open(json_file, "r") as f:
        predictions_cache = json.load(f)

    # Get the list of .npz files in the directory
    npz_files = [f for f in os.listdir(npz_directory) if f.endswith(".npz")]

    # Wrap tqdm around the list of files to process
    for filename in tqdm(npz_files):
        # Extract the key from the filename
        if prefix:
            key = filename.split(f"{prefix}")[-1].split(".npz")[0]
        else:
            key = filename.split(".npz")[0]

        # Check if key is present in JSON
        slice = None
        if key in predictions_cache:
            # Extract 'coords' and 'score'
            coords = predictions_cache[key].get("coords")
            score = predictions_cache[key].get("score")
            # check if 'slice' is present and add it to the dict
            if "slice" in predictions_cache[key]:
                slice = predictions_cache[key].get("slice")

            # Create a new JSON file with the same name as the key
            new_json_filename = f"{key}.json"
            with open(
                os.path.join(npz_directory, new_json_filename), "w"
            ) as new_json_file:
                if slice:
                    json.dump(
                        {"coords": coords, "score": score, "slice": slice},
                        new_json_file,
                    )
                else:
                    json.dump({"coords": coords, "score": score}, new_json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process npz files and associated JSON."
    )
    parser.add_argument(
        "-d", "--directory", required=True, help="Directory containing .npz files"
    )
    parser.add_argument(
        "-j",
        "--json_file",
        required=True,
        help="Path to JSON file with predictions cache",
    )
    parser.add_argument(
        "-p", "--prefix", default=None, help="Optional prefix for .npz files"
    )
    args = parser.parse_args()
    main(args.directory, args.json_file, args.prefix)
