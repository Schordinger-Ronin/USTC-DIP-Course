FILE=cityscapes
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/

# Create the target directory
mkdir -p $TARGET_DIR
echo "Downloading $URL dataset to $TARGET_DIR ..."

# Download the tar archive
curl -L $URL -o $TAR_FILE

# Extract to the ./datasets/ directory
tar -zxvf $TAR_FILE -C ./datasets/

# Remove the tar archive
rm $TAR_FILE

# Use >> to append new paths to the end of the existing files
find "${TARGET_DIR}train" -type f -name "*.jpg" | sort -V >> ./train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" | sort -V >> ./val_list.txt