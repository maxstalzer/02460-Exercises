# run_once_process.py
import torch
import numpy as np
import os

def read_images(path):
    with open(path, 'rb') as f:
        f.read(16)  # skip header
        return torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28).copy())

def read_labels(path):
    with open(path, 'rb') as f:
        f.read(8)  # skip header
        return torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8).copy()).long()

raw = 'data/MNIST/raw'
processed = 'data/processed'
os.makedirs(processed, exist_ok=True)

print("Processing training set...")
torch.save((
    read_images(f'{raw}/train-images-idx3-ubyte'),
    read_labels(f'{raw}/train-labels-idx1-ubyte')
), f'{processed}/training.pt')

print("Processing test set...")
torch.save((
    read_images(f'{raw}/t10k-images-idx3-ubyte'),
    read_labels(f'{raw}/t10k-labels-idx1-ubyte')
), f'{processed}/test.pt')

print("Done! Files saved to data/processed/")