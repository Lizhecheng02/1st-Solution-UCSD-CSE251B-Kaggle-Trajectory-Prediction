#!/bin/bash

hidden_dims=(64 128 256)
nheads=(2 4 8)
num_layers_list=(2 4 6 8)

for hidden_dim in "${hidden_dims[@]}"; do
  for nhead in "${nheads[@]}"; do
    for num_layers in "${num_layers_list[@]}"; do

      echo "Running experiments with hidden_dim=$hidden_dim, nhead=$nhead, num_layers=$num_layers"

      for fold in {0..7}; do
        gpu=$fold

        echo "  Starting fold $fold on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python train.py \
          --gpu $gpu \
          --n_folds 8 \
          --fold $fold \
          --model TransformerNet \
          --scale 1.0 \
          --batch_size 32 \
          --num_epochs 400 \
          --learning_rate 5e-4 \
          --weight_decay 1e-4 \
          --step_size 80 \
          --gamma 0.4 \
          --save_dir ./models/hd${hidden_dim}_nh${nhead}_nl${num_layers} \
          --input_dim 6 \
          --hidden_dim $hidden_dim \
          --output_dim 120 \
          --nhead $nhead \
          --num_layers $num_layers &

        sleep 5
      done

      wait
      echo "Finished hidden_dim=$hidden_dim, nhead=$nhead, num_layers=$num_layers"
    done
  done
done

echo "All experiments completed."
