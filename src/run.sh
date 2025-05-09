python train.py \
    --gpu 0 \
    --n_folds 10 \
    --fold 0 \
    --model TransformerNet \
    --scale 5.0 \
    --batch_size 32 \
    --num_epochs 400 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --step_size 80 \
    --gamma 0.4 \
    --save_dir ./models

