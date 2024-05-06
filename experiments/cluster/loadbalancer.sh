#!/usr/bin/env bash


combination=(
  # '1 2 5 6'
  '0 1'
)

qos_target=(
  '100'
)

policy=$1
gpuload=$2
hostname=$(hostname)

echo "$hostname working dir $(pwd)"
echo $gpuload

python main.py --task loadbalancer --model_num 2 --comb 0 1 --policy $policy --load 50 --qos 100 --queries 1000 --thld 5 --ways 2 --abandon --platform cluster --gpu 2080Ti --device 0  --predictor layer  --nodes 2 --gpuload $gpuload