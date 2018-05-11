#!/bin/bash 
mkdir -p saved/baseline/
wget -O saved/baseline/checkpoint.tar 'https://www.dropbox.com/s/ogq4r9m3ufw2zse/checkpoint.tar?dl=1'
wget -O saved/baseline/embedder.pkl 'https://www.dropbox.com/s/h5meab4u5z0zqz6/embedder.pkl?dl=1'
python3 caption_test.py --name baseline --checkpoint checkpoint.tar --data-dir $1 --output $2
