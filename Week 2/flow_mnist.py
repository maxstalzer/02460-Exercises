# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.3-MNIST (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

import argparse
# === NEW WORKING PATCH ===
from torchvision.datasets import MNIST

# Override the URLs directly. 
# This works for newer torchvision versions where 'mirrors' is ignored.
MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bb3b8fb87f'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]
# =========================

class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        # 1. Identify frozen (masked=1) and active (masked=0) parts
        z_frozen = z * self.mask
        z_active = z * (1 - self.mask)

        # 2. Compute scale (s) and translation (t) features
        # Masking the input ensures dependencies only on frozen parts
        s = self.scale_net(z_frozen)
        t = self.translation_net(z_frozen)

        # 3. Apply Affine Transformation
        x = z_frozen + (1 - self.mask) * (z_active * torch.exp(s) + t)

        # 4. Calculate Log Determinant
        log_det_J = torch.sum((1 - self.mask) * s, dim=1)
        
        return x, log_det_J
    
    def inverse(self, x):
        # 1. Identify frozen and active parts
        x_frozen = x * self.mask
        x_active = x * (1 - self.mask)

        # 2. Compute s and t
        s = self.scale_net(x_frozen)
        t = self.translation_net(x_frozen)

        # 3. Apply Inverse Affine Transformation
        z = x_frozen + (1 - self.mask) * (x_active - t) * torch.exp(-s)

        # 4. Inverse Log Determinant
        log_det_J = -torch.sum((1 - self.mask) * s, dim=1)

        return z, log_det_J

class Flow(nn.Module):
    def __init__(self, base, transformations):
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]
    
    def loss(self, x):
        return -torch.mean(self.log_prob(x))

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        for x, _ in data_loader: # MNIST loader returns (image, label)
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def get_chequerboard_mask(w, h, flip=False):
    """
    Creates a chequerboard mask of size (w, h).
    """
    mask = torch.zeros(h, w)
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == 0:
                mask[i, j] = 1
    if flip:
        mask = 1 - mask
    return mask.flatten() # Flatten to 1D for the linear layers

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model_mnist.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples_mnist.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--mask-type', type=str, default='chequerboard', choices=['random', 'chequerboard'], help='Masking strategy')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # --- 1. Load MNIST Data ---
    print("Loading MNIST...")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x + torch.rand(x.shape)/255), # Dequantize
                           transforms.Lambda(lambda x: x.flatten()) # Flatten to 784
                       ])),
        batch_size=args.batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                           transforms.Lambda(lambda x: x.flatten())
                       ])),
        batch_size=args.batch_size, shuffle=False)

    # --- 2. Define Model ---
    D = 784 # 28 * 28
    base = GaussianBase(D)
    
    transformations = []
    num_transformations = 8
    num_hidden = 512 # Needs to be larger for MNIST than for toy data

    for i in range(num_transformations):
        # --- Masking Strategy ---
        if args.mask_type == 'random':
            # Strategy 1: Random initialized masking
            mask = torch.randint(0, 2, (D,)).float()
        else:
            # Strategy 2: Chequerboard masking (inverted at each layer)
            mask = get_chequerboard_mask(28, 28, flip=(i % 2 == 1))

        # --- Network Definition ---
        scale_net = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, D),
            nn.Tanh() # <--- Stability fix requested in exercise
        )
        
        translation_net = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, D)
        )
        
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    model = Flow(base, transformations).to(args.device)

    # --- 3. Run Mode ---
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, args.device)
        torch.save(model.state_dict(), args.model)
        print(f"Model saved to {args.model}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        
        print("Generating samples...")
        with torch.no_grad():
            # Sample 64 images
            samples = model.sample((64,)).cpu()
            
            # Reshape from flattened (784) back to image (1, 28, 28)
            samples = samples.view(-1, 1, 28, 28)
            
            # Save the grid
            save_image(samples, args.samples, nrow=8)
            print(f"Samples saved to {args.samples}")