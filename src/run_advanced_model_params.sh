#!/bin/bash

hidden_dims=(64 128)
nheads=(4 8)
time_layers_list=(2 4 6)
agent_layers_list=(2 4 6)
gpu_ids=(3 4 5 6 7)

for hidden_dim in "${hidden_dims[@]}"; do
  for nhead in "${nheads[@]}"; do
    for time_layers in "${time_layers_list[@]}"; do
      for agent_layers in "${agent_layers_list[@]}"; do

        exp_name="hd${hidden_dim}_nh${nhead}_tl${time_layers}_al${agent_layers}"
        echo "==============================================================="
        echo "▶ 开始实验：$exp_name"
        echo "==============================================================="

        for fold in {0..9}; do
          idx=$(( fold % 5 ))
          gpu=${gpu_ids[$idx]}

          echo "  • Fold $fold  →  GPU $gpu"
          CUDA_VISIBLE_DEVICES=$gpu python train.py \
            --gpu $gpu \
            --n_folds 10 \
            --fold $fold \
            --model InteractionTransformer \
            --input_dim 31 \
            --hidden_dim $hidden_dim \
            --output_dim 120 \
            --nhead $nhead \
            --time_layers $time_layers \
            --agent_layers $agent_layers \
            --batch_size 32 \
            --num_epochs 400 \
            --learning_rate 3e-4 \
            --weight_decay 1e-4 \
            --step_size 80 \
            --gamma 0.4 \
            --save_dir "./models/${exp_name}/fold${fold}" &

          sleep 5
        done

        wait
        echo "✔ 完成实验：$exp_name"
        echo
      done
    done
  done
done

echo "🎉 全部实验已完成。"
