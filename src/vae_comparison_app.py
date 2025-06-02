import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# --- Generate toy time series data ---
@st.cache_data
def generate_data(seq_len=30, num_samples=500):
    t = np.linspace(0, 4 * np.pi, seq_len)
    data = [np.sin(t + np.random.rand()) + 0.1 * np.random.randn(seq_len) for _ in range(num_samples)]
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
    return (data - data.mean()) / data.std()  # Normalize

# --- VAE model for time series ---
class TimeSeriesVAE(nn.Module):
    def __init__(self, seq_len=30, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, seq_len),
            nn.Tanh()  # Output scaled between -1 and 1
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon.unsqueeze(-1), mu, logvar

# --- Transformer-based VAE ---
class TransformerTimeSeriesVAE(nn.Module):
    def __init__(self, seq_len=30, latent_dim=10):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, 16))
        self.input_proj = nn.Linear(1, 16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, seq_len),
            nn.Tanh()  # Output scaled
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        mu, logvar = self.mu(x), self.logvar(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon.unsqueeze(-1), mu, logvar

# --- Loss Function ---
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# --- Streamlit UI ---
st.title("VAE vs Transformer VAE on Toy Time Series")

model_choice = st.radio("Select Model", ("VAE", "Transformer VAE"))
epochs = st.slider("Training Epochs", 5, 200, 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
seq_len = 30
latent_dim = 10
data = generate_data(seq_len=seq_len).to(device)
train_data = data[:400]
test_data = data[400:]

# Initialize model
if model_choice == "VAE":
    model = TimeSeriesVAE(seq_len, latent_dim).to(device)
else:
    model = TransformerTimeSeriesVAE(seq_len, latent_dim).to(device)

# Train model
@st.cache_resource
def train(_model, _data, _epochs):
    optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)
    _model.train()
    for epoch in range(_epochs):
        optimizer.zero_grad()
        recon, mu, logvar = _model(_data)
        loss = vae_loss(recon, _data, mu, logvar)
        loss.backward()
        optimizer.step()
    return _model

model = train(model, train_data, epochs)

# Evaluate
model.eval()
with torch.no_grad():
    recon, mu, _ = model(test_data)
    recon = recon.squeeze(-1).cpu().numpy()
    original = test_data.squeeze(-1).cpu().numpy()
    mse = F.mse_loss(torch.tensor(recon), torch.tensor(original)).item()

# Plot reconstructions
def plot_sequences(orig, recon):
    fig, ax = plt.subplots(5, 1, figsize=(8, 6))
    for i in range(5):
        ax[i].plot(orig[i], label="Original")
        ax[i].plot(recon[i], label="Reconstructed")
        ax[i].legend()
    fig.tight_layout()
    return fig

st.pyplot(plot_sequences(original, recon))
st.write(f"🔎 **Reconstruction MSE:** `{mse:.4f}`")

# Plot latent space
with torch.no_grad():
    _, mu, _ = model(test_data)
    z = mu.cpu().numpy()
    z_2d = TSNE(n_components=2, random_state=42).fit_transform(z)

fig2, ax = plt.subplots()
ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6)
ax.set_title("Latent Space (t-SNE Projection)")
st.pyplot(fig2)

# Description
st.markdown("### Summary")
st.markdown(f"""
- You selected: **{model_choice}**
- Model trained on 400 samples of noisy sine waves.
- Reconstruction and latent space are shown.
- Use the **slider** to increase training epochs for better reconstructions.
""")
