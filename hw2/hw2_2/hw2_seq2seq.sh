#!/bin/bash 
mkdir -p saved/baseline/
wget -O saved/baseline/checkpoint.tar 'https://www.dropbox.com/s/9c4ptbtwuruh0je/checkpoint.tar?dl=1'
wget -O svaed/baseline/embedder.pkl 'https://www.dropbox.com/s/2ekvpgpmtn9x67z/embedder.pkl?dl=1'
python caption_test.py --name baseline --checkpoint checkpoint.tar --data-dir $1 --output $2
