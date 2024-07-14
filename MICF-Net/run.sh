python main.py --dataset 'Houston' --mode 'MICF-NET' --heads 8 --patch_size 9 --lr 2e-4 --dep 2 --lam1 0.3 --atte 8 --dim 64 --alpha 0.1

python main.py --dataset 'Muufl'   --mode 'MICF-NET' --heads 16 --patch_size 7 --lr 1e-3 --dep 2 --lam1 0.5 --atte 8 --dim 64 --alpha 0.1

python main.py --dataset 'Trento'  --mode 'MICF-NET' --heads 16 --patch_size 7 --lr 4e-4 --dep 2 --lam1 0.1 --atte 8 --dim 64 --alpha 0.1
