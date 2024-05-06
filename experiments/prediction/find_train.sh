#!/usr/bin/env bash
# bash experiments/7.2_qos/2in7.sh

# qos
CURRENT_DIR=$(cd "$(dirname "$0")";pwd)
from=0
test_len=1000

for ((i = from; i < test_len ; i++)); do
    python3 main.py --task train --platform single --gpu 2080Ti --device 0 --model_num 2 --mode all --modeling mlp  --model_comb all --predictor operator
    old_model_path="$CURRENT_DIR/../../model/2080Ti/2in7/all-operator.ckpt"
    new_model_path="$CURRENT_DIR/../../model/2080Ti/2in7/all-operator-$i.ckpt"
    cp -rf $old_model_path $new_model_path
    sleep 10
done