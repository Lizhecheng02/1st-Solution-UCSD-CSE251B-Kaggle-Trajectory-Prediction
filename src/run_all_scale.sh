#!/bin/bash

for scale in {1..7}
do
    scale_float=$(printf "%.1f" $scale)
    echo "Starting experiments for scale $scale_float"

    for fold in {0..9}
    do
        if [ $fold -lt 8 ]; then
            gpu=$fold
        else
            gpu=$((fold - 8))
        fi

        echo "  Starting fold $fold on GPU $gpu with scale $scale_float"
        CUDA_VISIBLE_DEVICES=$gpu python train.py \
            --gpu $gpu \
            --n_folds 10 \
            --fold $fold \
            --model TransformerNet \
            --scale $scale_float \
            --batch_size 32 \
            --num_epochs 400 \
            --learning_rate 5e-4 \
            --weight_decay 1e-4 \
            --step_size 80 \
            --gamma 0.4 \
            --save_dir ./models/val_mse_scale_$scale_float &

        sleep 5
    done

    wait
    echo "Completed all folds for scale $scale_float"
done

echo "All scale experiments completed."
