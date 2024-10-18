cd data
mkdir -p dynamic_replica
cd dynamic_replica

# Generate and loop through the list of URLs
for i in $(seq -w 000 085)
do
    # Construct the filename and URL
    filename="train_${i}.zip"
    url="https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/${filename}"

    # Download the zip file
    wget $url
    echo "Download of $filename completed"

    # Unzip the file
    unzip $filename
    echo "Unzipping of $filename completed"

    # Delete any directories ending with 'right'
    find . -maxdepth 1 -type d -name '*right' -exec rm -rf {} +

    # Delete the zip file
    rm $filename
    echo "Deletion of $filename completed"
done

# process the frame annotations
mv frame_annotations_train.jgz frame_annotations_train.gz
gunzip frame_annotations_train.gz
mv frame_annotations_train frame_annotations_train.json