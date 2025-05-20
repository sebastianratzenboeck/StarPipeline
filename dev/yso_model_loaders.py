import os
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from sedfitter.sed import SEDCube
from scipy.interpolate import NearestNDInterpolator, interp1d
from astropy.constants import c as c_light_speed
from sklearn.preprocessing import StandardScaler
from data import Star


class R17:
    def __init__(self, dir: str):
        self.dir = dir
        self.t_params = None
        self.sed_cube = None
        self.wave = None
        self.apertures = None
        self.X_features = None
        self.X_input = None
        self.y_output = None
        self.param_trafo = None
        self.feature_scaler = StandardScaler()
        self.interpolator = None
        # Define input columns and transformation functions
        # Stellar properties
        self.star_cols = ['star.radius', 'star.temperature', 'Source Luminosity', 'Av']
        self.star_cols_renamed = ['logR', 'logT', 'logL', 'Av']
        self.star_cols_trafo = {'logR': np.log10, 'logT': np.log10, 'logL': np.log10, 'Av': lambda x: x}
        # Disk properties
        self.disk_cols = ['inclination', 'disk.mass', 'disk.rmax', 'disk.beta', 'disk.p', 'disk.h100']
        self.disk_cols_renamed = ['incl', 'logDiskMass', 'logDiskRmax', 'DiskBeta', 'DiskP', 'logDiskH100']
        self.disk_cols_trafo = {
            'incl': lambda x: x, 'logDiskMass': np.log10, 'logDiskRmax': np.log10,
            'DiskBeta': lambda x: x, 'DiskP': lambda x: x, 'logDiskH100': np.log10
        }
        # Envelope and cavity properties
        self.envolpe_cavity_cols = ['envelope.rho_0', 'cavity.power', 'cavity.theta_0', 'cavity.rho_0']
        self.envolpe_cavity_cols_renamed = ['logEnvRho0', 'CavityPow', 'CavityTheta0', 'logCavityRho0']
        self.envolpe_cavity_cols_trafo = {
            'logEnvRho0': np.log10, 'CavityPow': lambda x: x, 'CavityTheta0': lambda x: x, 'logCavityRho0': np.log10
        }
        # Define the columns to keep
        self.all_cols = None
        self.all_cols_renamed = None
        self.all_cols_trafo = None
        self.X_features = None

    def postprocess(self):
        # Remove models which don't have any solutions
        cut = (self.t_params['star.radius'] != 0)
        cut_idx = np.where(cut)[0]
        # Remove the models which don't have any solutions
        self.t_params = self.t_params[cut_idx][self.all_cols].to_pandas()
        # Rename the columns to match the expected names
        self.t_params.rename(columns=dict(zip(self.all_cols, self.all_cols_renamed)), inplace=True)
        # Overwrite the SED cube with the numpy array
        self.y_output = self.sed_cube.val.value[cut_idx]
        # Post-process the parameters
        # Apply the transformation to the parameters
        for col, func in self.all_cols_trafo.items():
            if col in self.t_params.columns:
                self.t_params[col] = func(self.t_params[col])
        # Standardize features to zero one range
        self.X_input = self.t_params[self.X_features]
        self.feature_scaler.fit(self.X_input)
        X_interp = self.feature_scaler.transform(self.X_input)
        # Set up the interpolator
        self.interpolator = NearestNDInterpolator(X_interp, self.y_output, rescale=True)
        return self

    def transform_sed(self, f_nu, distance_pc):
        if isinstance(f_nu, np.ndarray):
            if not isinstance(f_nu, u.Quantity):
                f_nu = f_nu * u.mJy
        if isinstance(distance_pc, np.ndarray):
            if not isinstance(f_nu, u.Quantity):
                distance_pc = distance_pc * u.pc
        distance_kpc = distance_pc.to(u.kpc)
        # Convert to erg/s/cm²/Hz
        f_nu_cgs = f_nu.to(u.erg / u.s / u.cm**2 / u.Hz)
        # Convert wavelength to cm
        lam_cm = self.wave.to(u.cm)
        # Apply conversion
        f_lambda = (f_nu_cgs * c_light_speed) / lam_cm**2
        # Output in erg/s/cm²/Å (flam)
        f_lambda = f_lambda.to(u.erg / u.s / u.cm**2 / u.AA)
        # Factor in distance
        f_lambda *= (u.kpc / distance_kpc[...,None,None]) ** 2
        return f_lambda

    def load_model(self):
        params_path = os.path.join(self.dir, 'parameters.fits')
        flux_path = os.path.join(self.dir, 'flux.fits')
        self.t_params = Table.read(params_path)
        self.sed_cube = SEDCube.read(flux_path)
        self.wave = self.sed_cube.wav.to(u.AA)
        self.apertures = self.sed_cube.apertures
        # Call postprocess method
        self.postprocess()

    def sed(self, data: pd.DataFrame) -> tuple:
        """Get the SED for the given data"""
        # Get necessary parameters from kwargs
        distance_pc = data['distance'].values * u.pc
        X_data = self.feature_scaler.transform(data[self.X_features])
        f_nu = self.interpolator(X_data) * u.mJy
        # Transform to flux density in erg/s/cm²/AA
        f_lambda = self.transform_sed(f_nu, distance_pc)
        return f_lambda, self.wave

class SPUBHMI(R17):
    """Source embedded in ambient medium w/ density ρ_amb = 1e−23 g/cm3, with disk, envelope, and cavity
    Most complex model with all parameters
    """
    def __init__(self, dir: str):
        super().__init__(dir)
        self.all_cols = self.star_cols + self.disk_cols + self.envolpe_cavity_cols
        self.all_cols_renamed = self.star_cols_renamed + self.disk_cols_renamed + self.envolpe_cavity_cols_renamed
        self.all_cols_trafo = {**self.star_cols_trafo, **self.disk_cols_trafo, **self.envolpe_cavity_cols_trafo}
        self.X_features = [col for col in self.all_cols_renamed if col!='Av']
        # Load the model
        print("Loading SPUBHMI model...")
        self.load_model()
        print("SPUBHMI model loaded.")

class S___SMI(R17):
    """Source embedded in ambient medium w/ density ρ_amb = 1e−23 g/cm3, no disk, no envelope, no cavity
    Simplest model with only stellar parameters
    """
    def __init__(self, dir: str):
        super().__init__(dir)
        self.all_cols = [col for col in self.star_cols if col!='Av']
        self.all_cols_renamed = [col for col in self.star_cols_renamed if col!='Av']
        self.all_cols_trafo = {key: val for key, val in self.star_cols_trafo.items() if key!='Av'}
        self.X_features = self.all_cols_renamed
        # Load the model
        print("Loading S---MI model...")
        self.load_model()
        print("S---MI model loaded.")

class SP__HMI(R17):
    """Source embedded in ambient medium w/ density ρ_amb = 1e−23 g/cm3, with disk, no envelope, no cavity"""
    def __init__(self, dir: str):
        super().__init__(dir)
        self.all_cols = self.star_cols + self.disk_cols
        self.all_cols_renamed = self.star_cols_renamed + self.disk_cols_renamed
        self.all_cols_trafo = {**self.star_cols_trafo, **self.disk_cols_trafo}
        self.X_features = [col for col in self.all_cols_renamed if col!='Av']
        # Load the model
        print("Loading SP-HMI model...")
        self.load_model()
        print("SP-HMI model loaded.")

class SP__H_I(R17):
    """Source NOT embedded in medium, with disk, no envelope, no cavity"""
    def __init__(self, dir: str):
        super().__init__(dir)
        self.all_cols = self.star_cols + self.disk_cols
        self.all_cols_renamed = self.star_cols_renamed + self.disk_cols_renamed
        self.all_cols_trafo = {**self.star_cols_trafo, **self.disk_cols_trafo}
        self.X_features = [col for col in self.all_cols_renamed if col!='Av']
        # Load the model
        print("Loading SP-H-I model...")
        self.load_model()
        print("SP--H-I model loaded.")
