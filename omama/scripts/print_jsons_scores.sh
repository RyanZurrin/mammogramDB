#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <ids_file> <output_file> <data_dir>"
    exit 1
fi

ids_file="$1"
output_file="$2"
data_dir="$3"

# Clear out the output file for fresh content
> "$output_file"

while read -r id; do
    # Removing any potential white-space
    id=$(echo "$id" | tr -d '[:space:]')

    json_file="$data_dir/$id.json"
    # Log to console for debugging
    echo "Processing: $json_file"

    if [[ -f "$json_file" ]]; then
        # Extract the score value using jq
        score=$(jq .score "$json_file")

        # Write only the score to the output file
        echo "$score" >> "$output_file"
    else
        echo "File $json_file not found!"
    fi
done < "$ids_file"
