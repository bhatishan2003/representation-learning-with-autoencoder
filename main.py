import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict


# ======================
# Device
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# Model
# ======================
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, 1, 28, 28)
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
def train_model(args):

    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = AutoEncoder(args.latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    recon_losses = []
    ortho_losses = []

    model.train()

    for epoch in range(args.epochs):

        total_recon = 0
        total_ortho = 0

        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):

            x = x.to(DEVICE)
            optimizer.zero_grad()

            x_hat, z = model(x)
            recon_loss = F.mse_loss(x_hat, x)

            if args.use_ortho:
                ortho_loss = orthogonality_loss(z)
                loss = recon_loss + args.lambda_ortho * ortho_loss
                total_ortho += ortho_loss.item()
            else:
                loss = recon_loss

            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()

        recon_losses.append(total_recon / len(loader))
        ortho_losses.append(total_ortho / len(loader) if args.use_ortho else 0)

    return model, recon_losses, ortho_losses


# ======================
# Latent Extraction
# ======================
def extract_latents(model, samples_per_class=100):

    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    model.eval()
    class_latents = defaultdict(list)

    with torch.no_grad():
        for x, y in dataset:
            if len(class_latents[y]) < samples_per_class:
                x = x.unsqueeze(0).to(DEVICE)
                _, z = model(x)
                class_latents[y].append(z.cpu().numpy())

            if all(len(v) >= samples_per_class for v in class_latents.values()):
                break

    return class_latents


# ======================
# Argument Parser
# ======================
def get_args():

    parser = argparse.ArgumentParser(
        description="MNIST AutoEncoder with Optional Orthogonality Loss"
    )

    parser.add_argument(
        "--use_ortho",
        action="store_true",
        help="Enable orthogonality loss"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Latent vector dimension"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    parser.add_argument(
        "--lambda_ortho",
        type=float,
        default=0.01,
        help="Orthogonality loss weight"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    print("\nTraining Configuration:")
    print(args)

    model, recon_losses, ortho_losses = train_model(args)

    # Save model
    model_name = "model_with_ortho.pth" if args.use_ortho else "model_no_ortho.pth"
    torch.save(model.state_dict(), model_name)

    # Save losses
    prefix = "with_ortho" if args.use_ortho else "no_ortho"
    np.save(f"recon_{prefix}.npy", recon_losses)
    np.save(f"ortho_{prefix}.npy", ortho_losses)

    # Extract & save latents
    print("Extracting latent representations...")
    latents = extract_latents(model)
    np.save(f"latents_{prefix}.npy", latents)

    print("\nTraining complete. Files saved.")