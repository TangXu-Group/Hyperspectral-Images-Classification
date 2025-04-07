python SpiralMamba-main.py --tao 0.95 --mixup_prob 0.05 --exp_name demo --dataset IP --patch_size 15 \
&& \
python SpiralMamba-main.py --tao 0.9 --mixup_prob 0.05 --exp_name demo --dataset PU --patch_size 15 \
&& \
python SpiralMamba-main.py --tao 0.7 --mixup_prob 0.3 --exp_name demo --dataset HUS --patch_size 11 \
&& \
python SpiralMamba-main.py --tao 0.85 --mixup_prob 0.05 --exp_name demo --dataset HUS18_100 --patch_size 17
