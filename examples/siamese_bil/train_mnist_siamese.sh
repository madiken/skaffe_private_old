#!/usr/bin/env sh
ROOT=/home/eustinova/skaffe-private
TOOLS=$ROOT/build/tools

srun --gres=gpu:1  $TOOLS/caffe train --solver=$ROOT/examples/siamese/mnist_siamese_solver.prototxt 