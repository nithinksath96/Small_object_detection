#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=2 --node_rank=1 --master_addr="10.1.165.241" \
    --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}