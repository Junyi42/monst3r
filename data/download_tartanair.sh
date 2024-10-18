TARGET_DIR="tartanair"
mkdir -p "$TARGET_DIR"

python download_tartanair.py --output-dir $TARGET_DIR --rgb --only-left --depth --only-hard

# Find and unzip all zip files
find "$TARGET_DIR" -type f -name "*.zip" -print0 | while IFS= read -r -d '' zipfile; do
    # Get the directory of the zip file
    zipdir=$(dirname "$zipfile")

    echo "Unzipping $zipfile to $zipdir"
    # Unzip to the respective directory, automatically overwrite existing files
    unzip -o -q "$zipfile" -d "$zipdir"

    # Check if the unzip was successful
    if [ $? -eq 0 ]; then
        echo "Deleting $zipfile"
        # Delete the zip file
        rm "$zipfile"
    else
        echo "Failed to unzip $zipfile"
    fi
done