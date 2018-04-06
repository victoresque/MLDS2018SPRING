for i in 32 64 128 160 192 256 512 640 768 1024 2048;
do
    python3.6 main.py --batch-size $i --epochs 1000 --save-freq 100 --dataset cifar --arch deep --save-dir models/saved/1-3-bonus/CIFARn/$i --validation-split 0.1 --noise
    python3.6 main.py --batch-size $i --epochs 1000 --save-freq 100 --dataset cifar --arch deep --save-dir models/saved/1-3-bonus/CIFAR/$i --validation-split 0.1
done

