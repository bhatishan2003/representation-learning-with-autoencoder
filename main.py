import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import os

# ======================
# Device
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================
# Dataset Download and Processing
# =================================
def _get_mnist_dataset():
    # Loading MNIST dataset
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    return dataset


# ======================
# Model
# ======================
class ImageAutoEncoder(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim=16):
        super().__init__()
        self.input_shape = input_shape
        assert len(input_shape) == 2
        flat_image_size = input_shape[0] * input_shape[1]
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_image_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, flat_image_size), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, 1, self.input_shape[0], self.input_shape[1])
        return x_hat, z


# ======================
# Orthogonality Loss
# ======================
def orthogonality_loss(z):
    z = F.normalize(z, dim=1)
    gram = torch.matmul(z, z.T)
    identity = torch.eye(gram.size(0), device=z.device)
    return torch.norm(gram - identity)


# ======================
# Training Function
# ======================
def main(args):
    # Data Loaders
    dataset = _get_mnist_dataset()
    image_size = dataset[0][0][0].shape
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initiating model
    model = ImageAutoEncoder(input_shape=image_size, latent_dim=args.latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss trackers
    recon_losses = []
    ortho_losses = []

    # Train loop
    for epoch in range(args.epochs):
        total_recon = 0
        total_ortho = 0

        for x, _ in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(DEVICE)
            optimizer.zero_grad()
            x_hat, z = model(x)
            recon_loss = F.mse_loss(x_hat, x)
            ortho_loss = orthogonality_loss(z)

            if args.use_ortho:
                loss = recon_loss + args.lambda_ortho * ortho_loss
            else:
                loss = recon_loss

            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_ortho += ortho_loss.item()

        recon_losses.append(total_recon / len(loader))
        ortho_losses.append(total_ortho / len(loader))

    return model, recon_losses, ortho_losses


# ======================
# Latent Extraction
# ======================
def extract_latents(model, samples_per_class=100):
    class_latents = defaultdict(list)
    dataset = _get_mnist_dataset()

    with torch.no_grad():
        for x, y in dataset:
            y = int(y)

            if len(class_latents[y]) < samples_per_class:
                x = x.unsqueeze(0).to(DEVICE)
                _, z = model(x)
                z = z.squeeze(0).cpu().numpy()
                class_latents[y].append(z)

            if all(len(v) >= samples_per_class for v in class_latents.values()):
                break

    return dict(class_latents)


# ======================
# Argument Parser
# ======================
def get_args():
    parser = argparse.ArgumentParser(description="MNIST AutoEncoder with Optional Orthogonality Loss")

    parser.add_argument("--use_ortho", action="store_true", help="Enable orthogonality loss")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--latent_dim", type=int, default=16, help="Latent vector dimension")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--lambda_ortho", type=float, default=0.01, help="Orthogonality loss weight")

    parser.add_argument("--experiment_root_dir", type=str, default=os.path.join(os.getcwd(), "results"))
    args = parser.parse_args()
    run_name = "with_ortho" if args.use_ortho else "without_ortho"
    run_name += "_latent-" + str(args.latent_dim)
    if args.use_ortho:
        run_name += "_lambda-" + str(args.lambda_ortho)
    args.run_dir = os.path.join(args.experiment_root_dir, run_name)
    os.makedirs(args.run_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()

    print("\nTraining Configuration:")
    print(args)

    model, recon_losses, ortho_losses = main(args)

    # Save model and losses
    torch.save(model.state_dict(), os.path.join(args.run_dir, "model.pth"))
    prefix = "with_ortho" if args.use_ortho else "no_ortho"
    np.save(os.path.join(args.run_dir, "recon.npy"), recon_losses)
    np.save(os.path.join(args.run_dir, "ortho.npy"), ortho_losses)

    # Extract & save latents
    print("Extracting latent representations...")
    latents = extract_latents(model)
    np.save(os.path.join(args.run_dir, "latents.npy"), latents)

    print("\nTraining complete. Files saved.")
