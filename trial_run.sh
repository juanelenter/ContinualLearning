#!/bin/bash

SEED=0
BUFFER_SIZE=2000

DATASET=seq-mnist
python utils/main.py \
--seed $SEED --dataset $DATASET --model cally \
--buffer_size $BUFFER_SIZE --batch_size 32 --minibatch_size 32 --lr 0.01 --n_epochs 10 \
--wandb_project TRIAL_CL --wandb_name cally_${BUFFER_SIZE}_${SEED}

DATASET=seq-cifar10
python utils/main.py \
--seed $SEED --dataset $DATASET --model cally \
--buffer_size $BUFFER_SIZE --batch_size 32 --minibatch_size 32 --lr 0.1 --n_epochs 10 \
--wandb_project TRIAL_CL --wandb_name cally_${BUFFER_SIZE}_${SEED}

