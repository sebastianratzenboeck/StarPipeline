import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim



class SpectrumNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 256, 512), dropout=0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            # Add dropout
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpectrumPredictor:
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128), dropout=0.2, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SpectrumNet(input_dim, output_dim, hidden_dims, dropout).to(self.device)
        self.dataset = None
        self.input_mean = None
        self.input_std = None

    def _normalize_inputs(self, X):
        return (X - self.input_mean) / self.input_std

    def _fit_input_normalization(self, X_train):
        self.input_mean = X_train.mean(axis=0)
        self.input_std = X_train.std(axis=0)
        self.input_std = np.clip(self.input_std, 1e-6, None)

    def train_model(
            self, X_train, Y_train,
            X_val=None, Y_val=None,
            n_epochs=100, batch_size=64, learning_rate=1e-3, verbose=True):
        self._fit_input_normalization(X_train)
        X_train_norm = self._normalize_inputs(X_train)
        train_dataset = SpectralDataset(X_train_norm, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if (X_val is not None) and (Y_val is not None):
            X_val_norm = self._normalize_inputs(X_val)
            val_dataset = SpectralDataset(X_val_norm, Y_val, normalize=True,
                                          means=train_dataset.means, stds=train_dataset.stds)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        self.model.to(self.device)

        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(1, n_epochs + 1):
            total_loss, total_count = 0.0, 0
            for xb, yb, mask in train_loader:
                xb, yb, mask = xb.to(self.device), yb.to(self.device), mask.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = ((pred - yb) ** 2 * mask).sum() / mask.sum()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                total_count += xb.size(0)

            avg_train_loss = total_loss / total_count

            if val_loader:
                self.model.eval()
                val_loss, val_count = 0.0, 0
                with torch.no_grad():
                    for xb, yb, mask in val_loader:
                        xb, yb, mask = xb.to(self.device), yb.to(self.device), mask.to(self.device)
                        pred = self.model(xb)
                        loss = ((pred - yb) ** 2 * mask).sum()
                        val_loss += loss.item()
                        val_count += xb.size(0)
                avg_val_loss = val_loss / val_count
                self.model.train()
                if verbose:
                    print(f"Epoch {epoch:>3} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            elif verbose:
                print(f"Epoch {epoch:>3} | Train Loss: {avg_train_loss:.6f}")

            if epoch % 10 == 0:
                checkpoint_path = f"checkpoints/spectrum_predictor_epoch{epoch}.pt"
                self.save(checkpoint_path)

        self.model.eval()
        self.dataset = train_dataset

    def predict(self, X):
        X = (X - self.input_mean) / self.input_std
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            Y_pred = self.model(X).cpu()
        return Y_pred

    def predict_flux(self, X):
        pred_norm = self.predict(X)
        return self.dataset.denormalize(pred_norm)

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_mean': self.input_mean,
            'input_std': self.input_std
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.output_dim = checkpoint['output_dim']
        self.input_mean = checkpoint['input_mean']
        self.input_std = checkpoint['input_std']
        self.model = SpectrumNet(self.input_dim, self.output_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()


class SpectralDataset(Dataset):
    def __init__(self, X, Y, log_floor=1e-36, normalize=True, means=None, stds=None):
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)
            Input physical parameters (e.g., logR, logL, logT, ..., aperture)
        Y : np.ndarray, shape (N, D_spec)
            Output spectra (possibly with NaNs)
        log_floor : float
            Minimum flux value before log-scaling
        normalize : bool
            Whether to normalize the spectra per wavelength bin
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.log_floor = log_floor
        self.normalize = normalize

        # Handle NaNs
        self.mask = ~np.isnan(Y)
        self.Y_raw = np.where(self.mask, Y, log_floor)

        # Log10 scale
        self.Y_log = np.log10(np.clip(self.Y_raw, log_floor, None))

        # Normalize (per wavelength bin)
        if normalize:
            if (means is None) or (stds is None):
                self.means = np.sum(self.Y_log * self.mask, axis=0) / np.sum(self.mask, axis=0)
                self.stds = np.sqrt(np.sum(((self.Y_log - self.means) ** 2) * self.mask, axis=0) / np.sum(self.mask, axis=0))
                self.stds = np.clip(self.stds, 1e-6, None)
                self.Y_processed = (self.Y_log - self.means) / self.stds
            else:
                self.means = means
                self.stds = stds
                self.Y_processed = (self.Y_log - self.means) / self.stds

        self.Y = torch.tensor(self.Y_processed, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]

    def denormalize(self, Y_tensor):
        """Convert normalized log spectrum back to physical fluxes."""
        if isinstance(Y_tensor, torch.Tensor):
            Y_np = Y_tensor.detach().cpu().numpy()
        else:
            Y_np = Y_tensor

        log_spectrum = Y_np * self.stds + self.means
        flux_spectrum = 10 ** log_spectrum
        return flux_spectrum


def reshape_spectral_aperture_data(X_phys, spectra, aperture_values):
    """
    Parameters
    ----------
    X_phys : np.ndarray, shape (N, 3)
        Array of logR, logL, logT values.
    spectra : np.ndarray, shape (N, 20, 200)
        The spectra at each aperture.
    aperture_values : np.ndarray, shape (20,)
        The physical values of the apertures.

    Returns
    -------
    X_full : np.ndarray, shape (N*20, 4)
        Expanded input: [logR, logL, logT, aperture]
    Y_full : np.ndarray, shape (N*20, 200)
        Corresponding spectra for each input row.
    """
    N, A, D = spectra.shape
    assert X_phys.shape[0] == N
    assert A == len(aperture_values)

    # Repeat physical inputs for each aperture
    X_phys_repeated = np.repeat(X_phys, A, axis=0)  # shape (N*A, 3)

    # Tile aperture values (once for each N)
    aperture_tiled = np.tile(aperture_values, N)[:, None]  # shape (N*A, 1)

    # Combine inputs
    X_full = np.hstack([X_phys_repeated, aperture_tiled])  # shape (N*A, 4)

    # Flatten spectra: (N*A, 200)
    Y_full = spectra.reshape(N * A, D)

    return X_full, Y_full