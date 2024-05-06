#!/bin/bash

comb_name=(
  # "{\"0\":0.72,\"1\":0.28}"
  # "{\"0\":0.73,\"1\":0.27}"
  # "{\"0\":0.74,\"1\":0.26}"
  # "{\"0\":0.75,\"1\":0.25}"
  # "{\"0\":0.76,\"1\":0.24}"
  # "{\"0\":0.77,\"1\":0.23}"
  # "{\"0\":0.78,\"1\":0.22}"
  # "{\"0\":0.79,\"1\":0.21}"
  # "{\"0\":0.80,\"1\":0.20}"
  # "{\"0\":0.81,\"1\":0.19}"
  # "{\"0\":0.82,\"1\":0.18}"

  # "{\"0\":0.70,\"1\":0.30}"
  # "{\"0\":0.72,\"1\":0.28}"
  # "{\"0\":0.74,\"1\":0.26}"
  # "{\"0\":0.75,\"1\":0.25}"
  # "{\"0\":0.77,\"1\":0.23}"
  # "{\"0\":0.78,\"1\":0.22}"
  "{\"0\":0.80,\"1\":0.20}"
  "{\"0\":0.82,\"1\":0.18}"
  "{\"0\":0.84,\"1\":0.16}"
  "{\"0\":0.86,\"1\":0.14}"
  "{\"0\":0.88,\"1\":0.12}"
  "{\"0\":0.90,\"1\":0.10}"
  "{\"0\":0.92,\"1\":0.08}"
)

file_name=(
  # "0.72"
  # "0.73"
  # "0.74"
  # "0.75"
  # "0.76"
  # "0.77"
  # "0.78"
  # "0.79"
  # "0.80"
  # "0.81"
  # "0.82"

  # "0.70"
  # "0.72"
  # "0.74"
  # "0.75"
  # "0.77"
  # "0.78"
  "0.80"
  "0.82"
  "0.84"
  "0.86"
  "0.88"
  "0.90"
  "0.92"

)


load_len=${#comb_name[*]}
CURRENT_DIR=$(cd "$(dirname "$0")";pwd)

for ((i = 0; i < load_len; i++)); do
    echo ${comb_name[$i]}
    bash $CURRENT_DIR/setup.sh Abacus ${comb_name[$i]}
    sleep 10
    bash $CURRENT_DIR/stop.sh
    scp onceas@133.133.135.71:/home/onceas/wanna/mt-dnn/results/cluster/Abacus/resnet50resnet101.csv /home/onceas/wanna/mt-dnn/results/cluster/Abacus/resnet50resnet101-2.csv
    sleep 1

    bash $CURRENT_DIR/setup.sh Clock ${comb_name[$i]}
    sleep 10
    bash $CURRENT_DIR/stop.sh
    sleep 1
    scp onceas@133.133.135.71:/home/onceas/wanna/mt-dnn/results/cluster/Clock/resnet50resnet101.csv /home/onceas/wanna/mt-dnn/results/cluster/Clock/resnet50resnet101-2.csv

    ssh onceas@133.133.135.71 "mv /home/onceas/wanna/mt-dnn/results/cluster /home/onceas/wanna/mt-dnn/results/cluster-${file_name[$i]}"
    mv $CURRENT_DIR/../../results/cluster $CURRENT_DIR/../../results/cluster-${file_name[$i]}
done 
