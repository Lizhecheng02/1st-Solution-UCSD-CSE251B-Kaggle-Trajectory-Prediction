python train.py \
    --gpu 0 \
    --n_folds 10 \
    --fold 0 \
    --model InteractionTransformer \
    --scale 1.0 \
    --batch_size 32 \
    --num_epochs 400 \
    --learning_rate 2.5e-4 \
    --weight_decay 1e-4 \
    --step_size 80 \
    --gamma 0.5 \
    --save_dir ./models

