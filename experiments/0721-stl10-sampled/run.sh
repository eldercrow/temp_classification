export PYTHONPATH='/home/hyunjoon/github/dav_sc:${PYTHONPATH}'
export CUDA_VISIBLE_DEVICES='0,1'

# two gpus for training
# three cpu process for each gpu for data preprocessing
python ../../train.py --cfg config.yaml --gpus 2 --num-workers 3