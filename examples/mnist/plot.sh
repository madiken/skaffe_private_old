#!/usr/bin/env sh
ROOT=/home/eustinova/skaffe-private
python $ROOT/tools/extra/parse_log.py \
    /home/eustinova/skaffe-private/examples/mnist/log.txt \
    /home/eustinova/skaffe-private/examples/mnist/

    python $ROOT/tools/extra/parse_log.py \
    /home/eustinova/skaffe-private/examples/mnist_bil/log.txt \
    /home/eustinova/skaffe-private/examples/mnist_bil/

python $ROOT/tools/extra/plot_log.py \
    /home/eustinova/skaffe-private/examples/mnist/log.txt.$1 /home/eustinova/skaffe-private/examples/mnist_bil/log.txt.$1 -f $2  -f $2 -p maxpool -p bilpool
