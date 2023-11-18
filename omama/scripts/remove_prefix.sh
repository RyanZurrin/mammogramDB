#!/bin/bash

# Check for required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 /path/to/directory prefix_to_remove"
    exit 1
fi

# Directory containing the files
DIRECTORY="$1"
# Prefix to remove
PREFIX="$2"

# Initialize counter
count=0

# Get total number of matching files for progress tracking
total=$(ls -l "$DIRECTORY/$PREFIX"*.npz 2>/dev/null | wc -l)

# Traverse through each file in the directory
for old_file in "$DIRECTORY/$PREFIX"*.npz; do
    # Check if any files match the pattern
    if [ -f "$old_file" ]; then
        # Extract just the filename without the path
        old_file_name=$(basename -- "$old_file")

        # Remove prefix
        new_file_name=${old_file_name#"$PREFIX"}

        # Full path for the new filename
        new_file="$DIRECTORY/$new_file_name"

        # Rename the file
        mv -n "$old_file" "$new_file"

        # Update and display progress
        count=$((count + 1))
        echo -ne "Processed $count/$total files\r"
    else
        echo "No matching files found."
        break
    fi
done

# Newline for clean exit
echo -e "\nDone."
echo
