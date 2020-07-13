#!/bin/bash

docker build --network=host -t pytorch-cuda102-cudnn7 --file Dockerfile .
