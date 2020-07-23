export PYTHONPATH='/home/hyunjoon/github/dav_sc:${PYTHONPATH}'
export CUDA_VISIBLE_DEVICES='0,1'

python ../../test.py \
    --config config.yaml \
    --snapshot './logs/lightning_logs/version_11/checkpoints/epoch=96.ckpt'