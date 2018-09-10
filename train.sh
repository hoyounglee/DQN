#!/bin/sh

#python3 /home/hylee/reinforcement_learning/RainBow_Hylee/main.py --is_double 1 --is_duel 1 --is_per 1 --is_distributional 0 --tarin_step 100000 --is_noisy 0 --num_step 1 --summary_path ./summary/RainBow_1 --env SpaceInvaders-v0

python3 /home/hylee/reinforcement_learning/RainBow_Hylee/main.py --is_double 1 --is_duel 1 --is_per 1 --is_distributional 0 --tarin_step 100000  --is_noisy 1 --num_step 1 --summary_path ./summary/RainBow_2 --env SpaceInvaders-v0

python3 /home/hylee/reinforcement_learning/RainBow_Hylee/main.py --is_double 0 --is_duel 0 --is_per 0 --is_distributional 0 --tarin_step 100000  --is_noisy 0 --num_step 1 --summary_path ./summary/dqn --env SpaceInvaders-v0
