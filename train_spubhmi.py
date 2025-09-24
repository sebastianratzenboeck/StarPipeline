import sys
import numpy as np
from astropy import units as u
from sklearn.model_selection import train_test_split
sys.path.append('dev/')
from dev.yso_model_loaders import SPUBHMI
from dev.spectrum_net import reshape_spectral_aperture_data, SpectrumPredictor


def train_spubhmi_model(base_path_yso_data):
    # Load the SPUBHMI model
    yso_model = SPUBHMI(base_path_yso_data + 'spubhmi')
    # Create the SpectrumPredictor instance
    X_star = yso_model.X_input
    distance_pc = np.full(X_star.shape[0], fill_value=1000.) * u.pc
    aperture = yso_model.apertures
    # Transform the spectra to flam
    flam = yso_model.transform_sed(yso_model.y_output, distance_pc)
    # Reshape the data for training
    X_full, Y_full = reshape_spectral_aperture_data(X_star, flam.value, aperture.value)
    X_full[:, -1] = np.log10(X_full[:, -1])
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, Y_full, test_size=0.1, random_state=42, shuffle=True
    )
    predictor = SpectrumPredictor(
        input_dim=X_train.shape[-1], output_dim=Y_full.shape[-1],
        hidden_dims=(1024, 1024), dropout=0.1
    )
    print('Training SPUBHMI emulator model...')
    predictor.train_model(
        X_train, y_train,
        X_val=X_val, Y_val=y_val,
        n_epochs=100, batch_size=2 ** 13,
        outdir='checkpoints/', save_every_n=2, verbose=True
    )


if __name__ == '__main__':
    # Define the base path for the YSO data
    base_path_yso_data = '/Users/ratzenboe/Documents/work/data_local/spectal_models/yso_models/models_richardson24/'
    # Train the SPUBHMI model
    train_spubhmi_model(base_path_yso_data)
    print("Training completed successfully.")
