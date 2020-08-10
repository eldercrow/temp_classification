export PYTHONPATH='/home/hyunjoon/github/dav_sc:${PYTHONPATH}'
export CUDA_VISIBLE_DEVICES='0,1'

# evaluation w/ pytorch-lightning
python ../../train.py --cfg config.yaml --gpus 1 --num-workers 1 --evaluate

# evaluation w/o pytorch-lightning
# python ../../test.py \
#     --config config.yaml \
#     --snapshot './logs/lightning_logs/version_11/checkpoints/epoch=96.ckpt'