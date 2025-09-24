import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class FastInterpolatorNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dims=[64, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Training setup
def train_with_validation(X_train, y_train, X_val, y_val, epochs=1000, patience=50):
    """
    Training with early stopping based on validation loss
    """
    # Normalize data
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train)
    y_train_scaled = output_scaler.fit_transform(y_train)
    X_val_scaled = input_scaler.transform(X_val)
    y_val_scaled = output_scaler.transform(y_val)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)

    model = FastInterpolatorNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train_tensor)
        train_loss = criterion(train_pred, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}")

    return model, input_scaler, output_scaler


class FastNNInterpolator:
    def __init__(self, model, input_scaler, output_scaler):
        self.model = model.eval()
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def __call__(self, X):
        """Fast inference - can handle single points or batches"""
        # Handle single point input
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        # Normalize and predict
        X_scaled = self.input_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            y_scaled = self.model(X_tensor).numpy()

        # Denormalize outputs
        y_pred = self.output_scaler.inverse_transform(y_scaled)

        return y_pred.squeeze() if squeeze_output else y_pred