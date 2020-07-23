export PYTHONPATH='/home/hyunjoon/github/dav_sc:${PYTHONPATH}'
export CUDA_VISIBLE_DEVICES='0,1'

python ../../train.py --cfg config.yaml --gpus 2 --num-workers 3