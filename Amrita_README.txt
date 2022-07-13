use python 3.7 for this code.
For installing packages do python3.7 -m pip install <whatever>

Modified command from the original one on Github:

CUDA_VISIBLE_DEVICES=4 python3.7 -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Glister --model InceptionV3 --lr 0.1 -sp ./result --batch 128

Need to change the CUDA_VISIBLE_DEVICES argument. If I put only '4', it uses GPU number 4, not 4 GPUs.
Might need to try with a comma separated list.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3.7 -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 1 --workers 10 --optimizer SGD -se 5 --selection Glister --model InceptionV3 --lr 0.1 -sp ./result --batch 128

^^ this successfully uses the 1st 4 GPUs with ids 0,1,2,3 together.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3.7 -u main.py --fraction 0.1 --dataset MNIST --data_path ~/datasets --num_exp 1 --workers 10 --optimizer SGD -se 5 --selection Glister --model InceptionV3 --lr 0.1 -sp ./result --batch 128


CUDA_VISIBLE_DEVICES=0,4,5,6,7 python3.7 -u main.py --fraction 0.1 --dataset TinyImageNet --data_path ~/datasets --num_exp 1 --workers 10 --optimizer SGD -se 5 --selection Forgetting --model LeNet --lr 0.1 -sp ./result --batch 128
