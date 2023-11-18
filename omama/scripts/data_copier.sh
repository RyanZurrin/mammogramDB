## Script that is used to copy data which is specified in a text file of paths
## to a new location, keeping the same directory structure from a specified number
## of levels up.  This is useful for copying data from a temporary location to a
## permanent location, or for copying data from one location to another.
##
## Usage: data_copier.sh <path to text file of paths> <number of levels to keep> <new location>
##
## Example: data_copier.sh /home/omama/paths.txt 5 /home/omama/new_data
##
## This will move all the data specified in the text file to the new location,
## keeping the same directory structure from 3 levels up.  For example, if the
## text file contains the following paths:
##
## /home/omama/data/2012/01/01/01/01/01
## /home/omama/data/2012/01/01/01/01/02
## /home/omama/data/2012/01/01/01/01/03
## /home/omama/data/2012/01/01/01/01/04
##
## The data will be moved to the following locations:
##
## /home/omama/new_data/2012/01/01/01/01/01
## /home/omama/new_data/2012/01/01/01/01/02
## /home/omama/new_data/2012/01/01/01/01/03
## /home/omama/new_data/2012/01/01/01/01/04
##
## The script will create the new directory structure if it does not already exist.

progress_bar() {
    # Get the number of lines in the text file
    num_lines=$(wc -l $1 | awk '{print $1}')
    # Get the number of lines that have been processed
    num_processed=$(cat $1 | grep -c "DONE")
    # Calculate the percentage of lines that have been processed
    percent=$(echo "scale=2; $num_processed / $num_lines * 100" | bc)
    # Calculate the number of progress bar characters to display
    num_chars=$(echo "scale=0; $percent / 2" | bc)
    # Calculate the number of spaces to display
    num_spaces=$(echo "scale=0; 50 - $num_chars" | bc)
    # Display the progress bar
    printf "\r["
    for (( i=0; i<$num_chars; i++ )); do
        printf "#"
    done
    for (( i=0; i<$num_spaces; i++ )); do
        printf " "
    done
    printf "] $percent%%"
}

# Check that the correct number of arguments have been supplied
if [ $# -ne 3 ]; then
    echo "Usage: data_copier.sh <path to text file of paths> <number of levels to keep> <new location>"
    exit 1
fi

# Check that the text file exists
if [ ! -f $1 ]; then
    echo "The text file does not exist"
    exit 1
fi

# Check that the number of levels to keep is a number and is valid
if ! [[ $2 =~ ^[0-9]+$ ]]; then
    echo "The number of levels to keep must be a number"
    exit 1
elif [ $2 -gt 0 ]; then
    # Get the first path from the text file
    first_path=$(head -n 1 $1)
    # Get the number of levels in the path
    num_levels=$(echo $first_path | tr "/" "\n" | wc -l)
    # Check that the number of levels to keep is valid
    if [ $2 -gt $num_levels ]; then
        echo "The number of levels to keep is greater than the number of levels in the paths"
        exit 1
    fi
fi

# Check that the new location exists and if not create it
if [ ! -d $3 ]; then
    mkdir -p $3
fi

# Loop through the paths in the text file
while read path; do
    # Get the number of levels in the path
    num_levels=$(echo $path | tr "/" "\n" | wc -l)
    # Get the number of levels to keep
    num_levels_to_keep=$2
    # Get the number of levels to remove
    num_levels_to_remove=$(echo "$num_levels - $num_levels_to_keep" | bc)
    # Remove the number of levels to remove from the path
    new_path=$(echo $path | cut -d "/" -f $num_levels_to_remove-$num_levels)
    # Add the new location to the path
    new_path="$3/$new_path"
    # Create the new directory structure if it does not already exist
    if [ ! -d $(dirname $new_path) ]; then
        mkdir -p $(dirname $new_path)
    fi
    # Copy the data to the new location
    cp -r $path $new_path
    # Add "DONE" to the end of the line in the text file
    sed -i "s|$path|$path DONE|g" $1
    # Display the progress bar
    progress_bar $1
done < $1

# now remove the "DONE" from the end of the lines in the text file so it is back
# to the original state
sed -i "s| DONE||g" $1

echo " DONE"