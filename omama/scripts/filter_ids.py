import json
import argparse


def filter_ids_by_score(input_file, output_file, score_threshold):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        while True:
            id_line = infile.readline().strip()
            json_line = infile.readline().strip()
            infile.readline()  # Read and discard the empty line

            if not id_line or not json_line:
                break  # End of file

            # Extract ID from the first line
            id_ = id_line.replace("===== ", "").replace(".json =====", "").strip()

            # Parse JSON data from the second line
            data = json.loads(json_line)
            print(data)

            if data.get("score", 0) >= score_threshold:
                outfile.write(id_ + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Filter IDs by score from a text file."
    )
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("output_file", type=str, help="Output file path")
    parser.add_argument(
        "--score", type=float, default=0.5, help="Score threshold (default: 0.5)"
    )

    args = parser.parse_args()

    filter_ids_by_score(args.input_file, args.output_file, args.score)

    print("Filtering complete.")


if __name__ == "__main__":
    main()
