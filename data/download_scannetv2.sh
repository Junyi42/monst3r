#!/bin/bash

# Set the number of threads
THREADS=4

# Define the function to process each scene
process_scene() {
    scene_id=$(printf "%04d" $1)  # Format the scene ID
    filename="data/scannetv2/scene${scene_id}_00.sens"
    output_path="data/scannetv2/scene${scene_id}_00"

    # Run the data processing command
    python datasets_preprocess/scannet_sens_reader.py --filename $filename --output_path $output_path

    # Delete the .sens file
    rm -rf $filename
}

export -f process_scene  # Export the function for use by xargs

# Use seq -w to generate numbers in the range 0707-0806 and use xargs for multi-threading
seq -w 707 806 | xargs -n 1 -P $THREADS -I {} bash -c 'process_scene "$@"' _ {}

echo "All scenes have been processed."
