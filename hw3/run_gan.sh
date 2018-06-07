#!/bin/bash 
mkdir -p saved/baseline_gan/
# wget -O saved/baseline_gan/checkpoint.tar 'https://www.dropbox.com/s/mr4y85lucehzvzc/checkpoint-epoch100-loss-2.7593.pth.tar?dl=1'
python3 test_gan.py --name baseline_gan --checkpoint checkpoint.tar --output samples/gan.png
