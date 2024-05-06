#!/bin/bash


# combination='1 2 5 6'
combination='0 1'

qos_target='100'

policy=$1

# qos
echo "working dir $(pwd)"


python main.py --task scheduler --model_num 2 --comb $combination --policy $policy --load 50 --qos $qos_target --queries 1000 --thld 5 --ways 2 --abandon --predictor layer --platform cluster --gpu 2080Ti --device 0 --node 0 --nodes 2
