#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.
ROOT=/home/eustinova/skaffe-private
EXAMPLES=$ROOT/build/examples/siamese
DATA=$ROOT/data/mnist

echo "Creating leveldb..."

rm -rf $ROOT/examples/siamese/mnist_siamese_train_leveldb
rm -rf $ROOT/examples/siamese/mnist_siamese_test_leveldb

$EXAMPLES/convert_mnist_siamese_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    $ROOT/examples/siamese/mnist_siamese_train_leveldb
$EXAMPLES/convert_mnist_siamese_data.bin \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    $ROOT/examples/siamese/mnist_siamese_test_leveldb

echo "Done."
