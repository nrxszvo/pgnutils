OMP_NUM_THREADS=7 torchrun --standalone --nnodes 1 --nproc-per-node 1 train.py --ngpu 1
