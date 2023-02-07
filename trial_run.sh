#!/bin/bash

SEED=0
BUFFER_SIZE=2000

DATASET=seq-mnist
for MODEL in cally er
do
    python utils/main.py \
    --seed $SEED --dataset $DATASET --model $MODEL \
    --buffer_size $BUFFER_SIZE --batch_size 32 --minibatch_size 32 --lr 0.01 --n_epochs 20 \
    --wandb_project MNIST_CL --wandb_name ${MODEL}_${BUFFER_SIZE}_${SEED}
done

python utils/main.py \
--seed $SEED --dataset $DATASET --model sgd \
--batch_size 32 --lr 0.01 --n_epochs 20 \
--wandb_project MNIST_CL --wandb_name sgd_${SEED}

DATASET=seq-cifar10
for MODEL in cally er
do
    python utils/main.py \
    --seed $SEED --dataset $DATASET --model $MODEL \
    --buffer_size $BUFFER_SIZE --batch_size 32 --minibatch_size 32 --lr 0.1 --n_epochs 30 \
    --wandb_project CIFAR_CL --wandb_name ${MODEL}_${BUFFER_SIZE}_${SEED}
done

python utils/main.py \
--seed $SEED --dataset $DATASET --model sgd \
--batch_size 32 --lr 0.1 --n_epochs 30 \
--wandb_project CIFAR_CL --wandb_name sgd_${SEED}
