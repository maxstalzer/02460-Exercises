# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def latent_poly2_curve(t, w0, w1, w2):
    return w2 * t**2 + w1 * t + w0

def compute_manifold_length(curve_func, model, points=100, device='cpu'):
    model.eval()
    t = torch.linspace(0, 1, points).view(-1, 1).to(device)
    z = curve_func(t) 
    
    with torch.no_grad():
        logits = model.decoder.decoder_net(z)
        x_manifold = torch.sigmoid(logits) 
        
        # Flatten the (points, 28, 28) output back to (points, 784) for distance calc
        x_flat = x_manifold.view(points, -1)
        
    distances = torch.norm(x_flat[:-1] - x_flat[1:], dim=1)
    manifold_length = torch.sum(distances).item()
    
    return manifold_length, z.cpu().numpy(), x_manifold.cpu().numpy()

def plot_curve_and_manifold(model, data_loader, z_path, x_path, device):
    """
    Plots the latent curve over the encoded data points, and shows the decoded images.
    """
    # 1. Get latent embeddings for the background scatter plot
    model.eval()
    z_background = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            # Encode the batch to get the mean of the latent distribution
            mean, _ = torch.chunk(model.encoder.encoder_net(x), 2, dim=-1)
            z_background.append(mean.cpu())
            labels.append(y.cpu())
            if len(z_background) > 20: # Just grab a few batches for the background
                break
                
    z_background = torch.cat(z_background).numpy()
    labels = torch.cat(labels).numpy()

    # 2. Set up the plots
    fig = plt.figure(figsize=(12, 5))
    
    # --- Plot A: The Latent Space ---
    ax1 = fig.add_subplot(1, 2, 1)
    scatter = ax1.scatter(z_background[:, 0], z_background[:, 1], c=labels, cmap='tab10', alpha=0.3, s=10)
    # Draw the curve on top
    ax1.plot(z_path[:, 0], z_path[:, 1], color='red', linewidth=2, marker='o', markersize=3, label='Latent Curve c(t)')
    ax1.scatter(z_path[0, 0], z_path[0, 1], color='black', s=50, label='Start (t=0)', zorder=5)
    ax1.set_title("Latent Space (z)")
    ax1.legend()
    
    # --- Plot B: The Decoded Digits along the curve ---
    ax2 = fig.add_subplot(1, 2, 2)
    # We will pick 10 evenly spaced points along the curve to display
    num_images_to_show = 10
    indices = torch.linspace(0, len(x_path)-1, num_images_to_show).long().numpy()
    
    # Concatenate the images horizontally
    strip = torch.cat([torch.tensor(x_path[i]) for i in indices], dim=1)
    ax2.imshow(strip.numpy(), cmap='gray')
    ax2.set_title("Decoded Manifold Path f(c(t))")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('curve_plot.png')

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'curve'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--model', type=str, default='vae_model.pt', help='file to save or load model to/from (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'curve':

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Create an interesting curved path
        # A parabolic curve that hits the origin at exactly t=0.5
        w0 = torch.tensor([[-2.0, 2.0]]).to(device)  # Start top-left
        w1 = torch.tensor([[4.0, -8.0]]).to(device)  # Velocity pulls down and right
        w2 = torch.tensor([[0.0, 8.0]]).to(device)   # Curvature pulls back up

        my_curve = lambda t: latent_poly2_curve(t, w0, w1, w2)

        # Calculate length
        length, z_path, x_path = compute_manifold_length(my_curve, model, points=100, device=device)
        print(f"Manifold curve length: {length:.4f}")
        plot_curve_and_manifold(model, mnist_test_loader, z_path, x_path, device)
