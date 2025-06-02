import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------
# Step 1: Generate Sine Wave Data
# ----------------------------
def generate_sine_data(num_samples=1000, seq_len=50):
    x = np.linspace(0, 4 * np.pi, seq_len)
    data = []
    for _ in range(num_samples):
        phase = np.random.rand() * 2 * np.pi
        amplitude = np.random.rand() * 0.9 + 0.1
        wave = amplitude * np.sin(x + phase)
        data.append(wave)
    return np.array(data).astype(np.float32)

# Dataset
seq_len = 50
data = generate_sine_data(num_samples=1000, seq_len=seq_len)
data_tensor = torch.tensor(data).unsqueeze(-1)  # Shape: [N, seq_len, 1]

# ----------------------------
# Step 2: Transformer VAE Components
# ----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads):
        super().__init__()
        self.linear = nn.Linear(input_dim, model_dim)
        self.pe = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.linear(x)
        x = self.pe(x)
        return self.encoder(x)

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, model_dim, num_layers, num_heads, seq_len):
        super().__init__()
        self.latent_to_seq = nn.Linear(latent_dim, model_dim * seq_len)
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.pe = PositionalEncoding(model_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, z):
        x = self.latent_to_seq(z).view(-1, self.seq_len, self.model_dim)
        x = self.pe(x)
        x = self.decoder(x)
        return self.output_layer(x)

class TransformerVAE(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, latent_dim=16, num_layers=2, num_heads=4, seq_len=50):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_layers, num_heads)
        self.mean_layer = nn.Linear(model_dim * seq_len, latent_dim)
        self.logvar_layer = nn.Linear(model_dim * seq_len, latent_dim)
        self.decoder = TransformerDecoder(latent_dim, input_dim, model_dim, num_layers, num_heads, seq_len)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_out = self.encoder(x)  # [B, seq_len, model_dim]
        flat = enc_out.reshape(x.size(0), -1)
        mu = self.mean_layer(flat)
        logvar = self.logvar_layer(flat)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# ----------------------------
# Step 3: Loss Function
# ----------------------------
# ----------------------------
# Step 3: Loss Function with Free Bits
# ----------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0, free_bits=0.05):
    recon_loss = nn.MSELoss()(recon_x, x)

    # Compute per-dimension KL divergence
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.mean(kl_per_dim, dim=0)

    # Apply Free Bits trick
    kl_fb = torch.clamp(kl_per_dim, min=free_bits)
    kl_div = torch.sum(kl_fb)

    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss.item(), kl_div.item()


# ----------------------------
# Step 4: Training Loop
# ----------------------------
# ... [Your existing code above remains unchanged]

# ----------------------------
# Step 4: Training Loop with KL Annealing
# ----------------------------
# ----------------------------
# Step 4: Training Loop with KL Annealing and Free Bits
# ----------------------------
model = TransformerVAE(seq_len=seq_len)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
batch_size = 64

beta_start = 0.0
beta_max = 0.2  # cap β to avoid posterior collapse
free_bits = 0.05  # per-dimension KL floor

for epoch in range(epochs):
    beta = beta_start + (beta_max - beta_start) * (epoch / (epochs - 1))

    permutation = torch.randperm(data_tensor.size(0))
    total_loss, total_recon, total_kl = 0, 0, 0
    for i in range(0, data_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch = data_tensor[indices]

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)

        loss, recon_loss_val, kl_val = vae_loss(recon_batch, batch, mu, logvar, beta=beta, free_bits=free_bits)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss_val
        total_kl += kl_val

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Recon={total_recon:.4f}, KL={total_kl:.4f}, Beta={beta:.4f}")

# ----------------------------
# Step 5: Generate New Samples
# ----------------------------
model.eval()
with torch.no_grad():
    z = torch.randn(10, 16)  # Sample from standard normal
    generated = model.decoder(z).squeeze(-1).cpu().numpy()

# ----------------------------
# Step 6: Plot Generated Samples
# ----------------------------
plt.figure(figsize=(12, 6))
for i, wave in enumerate(generated):
    plt.plot(wave, label=f'Sample {i+1}')
plt.title("Generated Sinewave Samples (Transformer-VAE)")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
