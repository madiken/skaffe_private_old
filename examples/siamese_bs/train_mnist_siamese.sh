#!/usr/bin/env sh
ROOT=/home/eustinova/skaffe-private
TOOLS=$ROOT/build/tools

srun --gres=gpu:1  $TOOLS/caffe train --solver=/home/eustinova/skaffe-private/examples/siamese_bs/mnist_siamese_solver.prototxt
