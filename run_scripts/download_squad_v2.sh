#!/bin/bash

# Create the directory if it doesn't exist
mkdir -p squad_v2

# URLs of the files to download
train_url="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
dev_url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

# Download the files
echo "Downloading train-v2.0.json..."
curl -o ../data/squad_v2/train-v2.0.json $train_url

echo "Downloading dev-v2.0.json..."
curl -o ../data/squad_v2/dev-v2.0.json $dev_url

echo "Files have been downloaded and saved in the squad_v2 folder."