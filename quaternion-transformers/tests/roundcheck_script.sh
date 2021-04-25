#!/bin/bash

python train_and_eval.py --task sentiment --type real --num_heads 1 --num_layers 1 --d_model 8 --dff 8 --epochs 1
python train_and_eval.py --task sentiment --type quaternion --num_heads 1 --num_layers 1 --d_model 8 --dff 8 --epochs 1
python train_and_eval.py --task sentiment --type mixed --num_heads 1 --num_layers 1 --d_model 8 --dff 8 --epochs 1
python train_and_eval.py --task translation --type real --num_heads 1 --num_layers 1 --d_model 8 --dff 8 --epochs 1
python train_and_eval.py --task translation --type quaternion --num_heads 1 --num_layers 1 --d_model 8 --dff 8 --epochs 1
python train_and_eval.py --task translation --type mixed --num_heads 1 --num_layers 1 --d_model 8 --dff 8 --epochs 1