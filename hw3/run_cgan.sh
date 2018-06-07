#!/bin/bash 
mkdir -p saved/baseline_cgan/
wget -O saved/baseline_cgan/checkpoint.tar 'https://www.dropbox.com/s/6nn1rpdhn2ha8dx/checkpoint-epoch160-loss-2.7784.pth.tar?dl=1'
wget -O saved/baseline_cgan/embedder.pkl 'https://www.dropbox.com/s/qj12ll7jvkqdfp4/embedder.pkl?dl=1'
python3 test_cgan.py --name baseline_cgan --checkpoint checkpoint.tar --input $1 --output samples/cgan.png
