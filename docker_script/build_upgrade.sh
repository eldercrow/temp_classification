#!/bin/bash

docker build --network=host --file ./Dockerfile_update -t pytorch-cuda102-cudnn7 .
