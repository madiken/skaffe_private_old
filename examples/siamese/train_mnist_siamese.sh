#!/usr/bin/env sh

TOOLS=../../build/tools

srun --gres=gpu:1 $TOOLS/caffe train --solver=../../examples/siamese/mnist_siamese_solver.prototxt
