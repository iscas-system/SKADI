#!/usr/bin/env bash

policy=$1
gpuload=$2

CURRENT_DIR=$(cd "$(dirname "$0")";pwd)
nohup bash $CURRENT_DIR/server.sh $policy >/dev/null 2>&1 &
ssh onceas@133.133.135.71 "cd wanna/mt-dnn ; source /home/onceas/anaconda3/bin/activate abacus37 ; nohup bash ./experiments/cluster/server.sh $policy >/dev/null 2>&1 &"
sleep 40
nohup bash $CURRENT_DIR/loadbalancer.sh $policy $gpuload >/dev/null 2>&1 &
sleep 5
bash $CURRENT_DIR/client.sh $policy 

