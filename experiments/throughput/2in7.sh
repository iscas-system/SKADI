#!/usr/bin/env bash


combination=(
  '0 1' # resnet50 resnet101
  '0 3' # resnet50 incep
  '4 7' # resnet34 vgg16
  '5 7' # resnet34 vgg19
)

# combination=(
#   '0 1'
#   '0 2'
#   '0 3'
#   '0 4'
#   '0 5'
#   '0 7'
#   '1 2'
#   '1 3'
#   '1 4'
#   '1 5'
#   '1 7'
#   '2 3'
#   '2 4'
#   '2 5'
#   '2 7'
#   '3 4'
#   '3 5'
#   '3 7'
#   '4 5'
#   '4 7'
#   '5 7'
# )

qos_target=(
  '100' # Res50+Res101
  # '150' # Res50+Res152
  '100' # Res50+IncepV3
  # '50' # Res50+VGG16
  # '50' # Res50+VGG19
  # '50' # Res50+Resnet34
  # '160' # Res101+Res152
  # '150' # Res101+IncepV3
  # '90' # Res101+VGG16
  # '80' # Res101+VGG19
  # '100' # Res101+Resnet34
  # '150' # Res152+IncepV3
  # '150' # Res152+VGG16
  # '150' # Res152+VGG19
  # '150' # Res152+Resnet34
  # '80' # IncepV3+VGG16
  # '80' # IncepV3+VGG19
  # '80' # IncepV3+Resnet34
  # '50' # VGG16+VGG19
  '50' # VGG16+Resnet34
  '60' # VGG19+Resnet34
)


CURRENT_DIR=$(cd "$(dirname "$0")";pwd)
from=0
load_len=${#load[*]}
method_len=${#comb_name[*]}
comb_len=${#combination[*]}

res_dir="$CURRENT_DIR/../../results/2080Ti/2in7"
echo $res_dir

for ((i = from; i < load_len; i++)); do
  for ((k = 0; k < comb_len; k++)); do
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy Abacus --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --abandon --predictor layer
    sleep 15
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy linear --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --abandon --predictor layer
    sleep 15
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy mt-dnn --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --abandon --predictor layer
    sleep 15
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy tcp --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --abandon --predictor layer
    sleep 15
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy FCFS --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --abandon --predictor layer
    sleep 15
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy SJF --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --abandon --predictor layer
    sleep 15
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy EDF --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --abandon --predictor layer
    sleep 15
  done
  python3 ./experiments/7.2_qos/percentile2.py
  sleep 2
  # tag=`echo ${combination["$k"]} | sed s/' '//g`
  new_res_dir="$CURRENT_DIR/../../results/2080Ti/2in7-load${load["$i"]}"
  echo $new_res_dir
  cp -rf $res_dir $new_res_dir
done


# for ((i = tested_comb; i < comb_len; i++)); do
#   python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy SJF --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
# done

# for ((i = tested_comb; i < comb_len; i++)); do
#   python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
# done

# for ((i = tested_comb; i < comb_len; i++)); do
#   python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy EDF --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
# done

# for ((i = tested_comb; i < comb_len; i++)); do
#   python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
# done

# cp -r results/A100/2in7 data/server/7.3_throughput/
# throughput
# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy SJF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy EDF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done
