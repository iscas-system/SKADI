#!/usr/bin/env bash
# bash experiments/7.2_qos/2in7.sh

combination=(
  '0 1'
#   '0 2'
#   '0 3'
#   '0 4'
#   '0 5'
#   '1 2'
#   '1 3'
#   '1 4'
#   '1 5'
#   '2 3'
#   '2 4'
#   '2 5'
#   '3 4'
#   '3 5'
#   '4 5'
)

comb_name=(
  "resnet50resnet101.csv"
#   "resnet50resnet152.csv"
#   "resnet50inception_v3.csv"
#   "resnet50vgg16.csv"
#   "resnet50vgg19.csv"
#   "resnet101resnet152.csv"
#   "resnet101inception_v3.csv"
#   "resnet101vgg16.csv"
#   "resnet101vgg19.csv"
#   "resnet152inception_v3.csv"
#   "resnet152vgg16.csv"
#   "resnet152vgg19.csv"
#   "inception_v3vgg16.csv"
#   "inception_v3vgg19.csv"
#   "vgg16vgg19.csv"
)

comb_name2=(
  "resnet50resnet101-2.csv"
#   "resnet50resnet152.csv"
#   "resnet50inception_v3.csv"
#   "resnet50vgg16.csv"
#   "resnet50vgg19.csv"
#   "resnet101resnet152.csv"
#   "resnet101inception_v3.csv"
#   "resnet101vgg16.csv"
#   "resnet101vgg19.csv"
#   "resnet152inception_v3.csv"
#   "resnet152vgg16.csv"
#   "resnet152vgg19.csv"
#   "inception_v3vgg16.csv"
#   "inception_v3vgg19.csv"
#   "vgg16vgg19.csv"
)

# qos
CURRENT_DIR=$(cd "$(dirname "$0")";pwd)
method_len=${#comb_name[*]}
comb_len=${#combination[*]}

res_dir="$CURRENT_DIR/../../results/2080Ti/2in7"
echo $res_dir

for ((k = 0; k < comb_len; k++)); do
    python3 main.py --task profile --platform single --gpu 2080Ti --device 0 --test 500  --model_num 2
    # sleep 5
    # old_res_path="$CURRENT_DIR/../../results/2080Ti/2in7/comb_name["$k"]"
    # new_res_path="$CURRENT_DIR/../../results/2080Ti/2in7/comb_name2["$k"]"
    # cp -f $old_res_path $new_res_dir

    # python3 main.py --task profile --platform single --gpu 2080Ti --device 0 --test 500  --model_num 1
    # sleep 5
done
