#!/bin/bash

# wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
#     --output-document train-images-idx3-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
#     --output-document train-labels-idx1-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
#     --output-document t10k-images-idx3-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz \
#     --output-document t10k-labels-idx1-ubyte.gz

#!/bin/bash

# 使用 CVDF 镜像源替换原始 404 链接
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

wget "${BASE_URL}/train-images-idx3-ubyte.gz" -O train-images-idx3-ubyte.gz
wget "${BASE_URL}/train-labels-idx1-ubyte.gz" -O train-labels-idx1-ubyte.gz
wget "${BASE_URL}/t10k-images-idx3-ubyte.gz" -O t10k-images-idx3-ubyte.gz
wget "${BASE_URL}/t10k-labels-idx1-ubyte.gz" -O t10k-labels-idx1-ubyte.gz

# 下载后建议解压，因为 MGPUSim 的数据加载器通常读取解压后的二进制文件
# gunzip *.gz