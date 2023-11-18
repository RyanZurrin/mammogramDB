import re
import argparse


def extract_and_save(input_file, output_file):
    pattern = re.compile(r"===== (.*?)\.json =====")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            match = pattern.search(line)
            if match:
                outfile.write(match.group(1) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract identifiers from a text file."
    )
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("output_file", type=str, help="Output file path")

    args = parser.parse_args()

    extract_and_save(args.input_file, args.output_file)

    print("Extraction complete.")


if __name__ == "__main__":
    main()
