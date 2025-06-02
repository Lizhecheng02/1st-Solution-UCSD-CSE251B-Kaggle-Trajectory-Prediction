#!/bin/bash

for fold in {0..9}
do
    if [ $fold -lt 8 ]; then
        gpu=$fold
    else
        gpu=$((fold - 8))
    fi

    echo "Starting fold $fold on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --gpu $gpu \
        --n_folds 10 \
        --fold $fold \
        --model InteractionTransformer \
        --scale 1.0 \
        --batch_size 32 \
        --num_epochs 400 \
        --learning_rate 2.5e-4 \
        --weight_decay 1e-4 \
        --step_size 80 \
        --gamma 0.5 \
        --save_dir ./models &

    sleep 5
done

wait
echo "All training processes completed."
