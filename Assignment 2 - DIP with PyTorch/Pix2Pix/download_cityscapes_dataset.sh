FILE=cityscapes
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/

# 1. 创建目标文件夹
mkdir -p $TARGET_DIR
echo "Downloading $URL dataset to $TARGET_DIR ..."

# 2. 下载压缩包
curl -L $URL -o $TAR_FILE

# 3. 解压到 ./datasets/ 目录下
tar -zxvf $TAR_FILE -C ./datasets/

# 4. 删除压缩包
rm $TAR_FILE

# 5. 【核心修改】：使用 >> 将新路径追加到原有文件末尾
find "${TARGET_DIR}train" -type f -name "*.jpg" | sort -V >> ./train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" | sort -V >> ./val_list.txt