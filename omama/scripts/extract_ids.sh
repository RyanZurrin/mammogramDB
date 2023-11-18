#!/bin/bash

input_file="$1"
output_file="$2"

awk -F'/' '{print $NF}' "$input_file" | awk -F'DXm.' '{print $2}' > "$output_file"
