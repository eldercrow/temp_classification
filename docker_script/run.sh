#!/bin/bash

USERNAME=`echo $USER`

docker run --rm --gpus "all" -it \
  -e LOCAL_USER_ID=`id -u $USER` \
  -e LOCAL_USER_NAME=`id -u -n $USER` \
  -e LOCAL_GROUP_ID=`id -g $USER` \
  -e LOCAL_GROUP_NAME=`id -g -n $USER` \
  -v /home/hyunjoon/github:/home/$USERNAME/github \
  -v /home/hyunjoon/dataset:/home/$USERNAME/dataset \
  -v /home/hyunjoon/.local:/home/$USERNAME/.local \
  -v /home/hyunjoon/.cache:/home/$USERNAME/.cache \
  --network host \
  pytorch-cuda102-cudnn7 \
  /bin/bash
