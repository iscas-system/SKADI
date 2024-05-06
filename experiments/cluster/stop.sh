#!/bin/bash

for pid in $(ps -ef | grep python | grep main | awk '{print $2}'); do
  kill $pid
done

CURRENT_DIR=$(cd "$(dirname "$0")";pwd)
ssh onceas@133.133.135.71 "bash $CURRENT_DIR/stop.sh"