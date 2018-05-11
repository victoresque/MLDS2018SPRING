#!/bin/bash 
mkdir -p saved/baseline/
wget -O saved/baseline/checkpoint.tar 'https://www.dropbox.com/s/9c4ptbtwuruh0je/checkpoint.tar?dl=1'
wget -O saved/baseline/embedder.pkl 'https://www.dropbox.com/s/2ekvpgpmtn9x67z/embedder.pkl?dl=1'
python3 chatbot_test.py --name baseline --checkpoint checkpoint.tar --input $1 --output $2
