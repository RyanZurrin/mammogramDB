import os
import argparse


def remove_files_from_list(_file_list_path):
    # Read the file paths
    with open(_file_list_path, "r") as f:
        lines = f.readlines()

    # Remove the files
    for line in lines:
        file_path = line.strip()  # Remove any trailing whitespaces or newlines
        try:
            os.remove(file_path)
            print(f"Successfully removed {file_path}")
        except FileNotFoundError:
            print(f"File {file_path} not found")
        except PermissionError:
            print(f"Permission error while trying to remove {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove files listed in a text file.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the text file containing file paths to remove.",
    )

    args = parser.parse_args()
    file_list_path = args.file

    remove_files_from_list(file_list_path)
