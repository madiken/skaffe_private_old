#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist_adaptation/lenet_solver.prototxt --gpu=1

# /usr/local/cuda/bin/cuda-gdb --args ./build/tools/caffe train --solver=examples/mnist_adaptation/lenet_solver.prototxt --gpu=1
