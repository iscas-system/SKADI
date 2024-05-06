#!/bin/bash


combination=(
  # '1 2 5 6'
  '0 1'
)

qos_target=(
  '100'
)

policy=$1
hostname=$(hostname)

node_id=0
if [ $hostname == "dell04" ]; then
  node_id=0
elif [ $hostname == "dell01" ]; then
  node_id=1
else
  node_id=0
fi

echo "$hostname nodeid=$node_id, working dir $(pwd)"

python main.py --task server --model_num 2 --comb 0 1 --policy $policy --load 50 --qos ${qos_target} --queries 1000 --thld 5 --ways 2 --abandon --platform cluster --gpu 2080Ti --device 0 --node $node_id --predictor layer
