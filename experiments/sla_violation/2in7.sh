#!/usr/bin/env bash
# bash experiments/7.2_qos/2in7.sh

combination=(
  # '0 1' # resnet50 resnet101
  # '0 3' # resnet50 incep
  # '3 5' # incep vgg19
  # '0 5' # resnet50 vgg19
  # '4 7' # resnet34 vgg16
  # '5 7' # resnet34 vgg19
  # '1 2' # resnet101_resnet152
  '4 5' # vgg16_vgg19
)

qos_target=(
  # '100' # Res50+Res101
  # '150' # Res50+Res152
  # '100' # Res50+IncepV3
  # '50' # Res50+VGG16
  '60' # Res50+VGG19
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
  # '50' # VGG16+Resnet34
  # '60' # VGG19+Resnet34
)

comb_name=(
  'resnet101resnet152.csv'
  'vgg16vgg19.csv'
)

load=(
  '5'
  '10'
  '20'
  '30'
  '40'
  '50'
  '60'
  '70'
  '80'
  '90'
  '100'
  '110'
  '120'
  '130'
  '140'
  '150'
  '160'
)

# qos
CURRENT_DIR=$(cd "$(dirname "$0")";pwd)
from=0
load_len=${#load[*]}
method_len=${#comb_name[*]}
comb_len=${#combination[*]}

res_dir="$CURRENT_DIR/../../results/2080Ti/2in7"
echo $res_dir

# for ((i = from; i < load_len; i++)); do
#   for ((k = 0; k < comb_len; k++)); do
#     python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy Abacus --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --predictor layer
#     sleep 20
#     python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy linear --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --predictor layer
#     sleep 20
#     python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy mt-dnn --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --predictor layer
#     sleep 20
#     python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy tcp --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --predictor layer
#     sleep 20
#     python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy FCFS --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --predictor layer
#     sleep 20
#     python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy SJF --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --predictor layer
#     sleep 20
#     python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy EDF --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --predictor layer
#     sleep 20
#     python3 ./experiments/7.2_qos/percentile2.py
#     sleep 5
#   done
#   # tag=`echo ${combination["$k"]} | sed s/' '//g`
#   new_res_dir="$CURRENT_DIR/../../results/2080Ti/2in7-load${load["$i"]}"
#   echo $new_res_dir
#   cp -rf $res_dir $new_res_dir
# done


for ((i = from; i < load_len; i++)); do
  for ((k = 0; k < comb_len; k++)); do
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy Abacus --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --predictor layer
    sleep 20
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy linear --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --predictor layer
    sleep 20
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy mt-dnn --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --predictor layer
    sleep 20
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy tcp --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --predictor layer
    sleep 20
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy FCFS --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --predictor layer
    sleep 20
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy SJF --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --predictor layer
    sleep 20
    python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb ${combination["$k"]} --policy EDF --load ${load["$i"]} --qos ${qos_target["$k"]} --queries 1000 --thld 5 --ways 2 --predictor layer
    sleep 20
    # python3 ./experiments/7.2_qos/percentile2.py
    sleep 5
  done
  # tag=`echo ${combination["$k"]} | sed s/' '//g`
  new_res_dir="$CURRENT_DIR/../../results/2080Ti/2in7-load${load["$i"]}"
  echo $new_res_dir
  for ((j = 0; j < method_len; j++)); do
    cp -rf $res_dir/mt-dnn/${comb_name["$j"]} $new_res_dir/mt-dnn/${comb_name["$j"]}
    cp -rf $res_dir/EDF/${comb_name["$j"]} $new_res_dir/EDF/${comb_name["$j"]}
    cp -rf $res_dir/FCFS/${comb_name["$j"]} $new_res_dir/FCFS/${comb_name["$j"]}
    cp -rf $res_dir/SJF/${comb_name["$j"]} $new_res_dir/SJF/${comb_name["$j"]}
    cp -rf $res_dir/linear/${comb_name["$j"]} $new_res_dir/linear/${comb_name["$j"]}
    cp -rf $res_dir/tcp/${comb_name["$j"]} $new_res_dir/tcp/${comb_name["$j"]}
    cp -rf $res_dir/mt-dnn/${comb_name["$j"]} $new_res_dir/mt-dnn/${comb_name["$j"]}
  done
done


# for ((i = 0; i < load_len; i++)); do
#   python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb 0 1 --policy Abacus --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --abandon
#   python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb 0 1 --policy FCFS --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --abandon
#   python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb 0 1 --policy SJF --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --abandon
#   python3 main.py --task server --platform single --node 0 --gpu 2080Ti --device 0  --model_num 2 --comb 0 1 --policy EDF --load ${load["$i"]} --qos 50 --queries 1000 --thld 5 --ways 2 --abandon
#   new_res_dir="$CURRENT_DIR/../../results/2080Ti/2in7-01-load${load["$i"]}-ab"
#   cp -r $res_dir $new_res_dir
#   sleep 10
# done

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
# cp -r scripts/server/7.2_qos/2in7 data/server/7.2_qos