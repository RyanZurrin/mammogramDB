import os
import argparse


def load_ids(file_path):
    with open(file_path, "r") as file:
        return set(line.strip() for line in file)


def find_matching_files(root_dir, ids_set):
    matching_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(id_ in file for id_ in ids_set):
                full_path = os.path.join(root, file)
                matching_paths.append(full_path)
    return matching_paths


def write_paths_to_file(paths, output_file):
    with open(output_file, "w") as file:
        for path in paths:
            file.write(path + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find and list full paths of DICOM images."
    )
    parser.add_argument("ids_file", type=str, help="Text file containing IDs")
    parser.add_argument("output_file", type=str, help="Output file to save paths")
    args = parser.parse_args()

    ids_set = load_ids(args.ids_file)
    root_dir = "/raid/data01/deephealth"
    matching_paths = find_matching_files(root_dir, ids_set)
    write_paths_to_file(matching_paths, args.output_file)

    print(
        f"Found {len(matching_paths)} matching files. Paths written to {args.output_file}"
    )


if __name__ == "__main__":
    main()
