OMP_NUM_THREADS=24 torchrun --standalone --nnodes 1 --nproc-per-node 1 train.py --ngpu 1 --train_heads --core_ckpt ~/mimicChessData/models/v0.1/20241113220006.ckpt --cfg cfghead.yml
