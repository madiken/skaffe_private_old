#!/usr/bin/env sh
ROOT=/home/eustinova/skaffe-private
python $ROOT/tools/extra/parse_log.py \
    /home/eustinova/skaffe-private/examples/siamese/log.txt \
    /home/eustinova/skaffe-private/examples/siamese/

python $ROOT/tools/extra/plot_log.py \
    /home/eustinova/skaffe-private/examples/siamese/log.txt.$1 -f $2 
