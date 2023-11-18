#!/bin/bash

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <ids_file> <output_file> <data_dir> <threshold>"
    exit 1
fi

ids_file="$1"
output_file="$2"
data_dir="$3"
threshold="$4"

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

        # Check if the score is less than the threshold
        if (( $(echo "$score < $threshold" | bc -l) )); then
            echo "===== $id.json =====" >> "$output_file"
            cat "$json_file" >> "$output_file"
            echo -e "\n" >> "$output_file"
        else
            continue
        fi
    else
        echo "File $json_file not found!"
    fi
done < "$ids_file"
