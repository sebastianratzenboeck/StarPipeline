import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from base import PipelineStep
from data import Star


class EvoTrackPARSEC(PipelineStep):
    def __init__(self, evo_track_path: str):
        self.df = pd.read_csv(evo_track_path)
        self.interpolator = self.init_interpolator()

    def init_interpolator(self):
        # Create a linear interpolator for logL, logT, and logR
        # based on logAge, logMass, and logZ
        # interp_nn =  LinearNDInterpolator(
        # Fit in nearest nd interpolator is much faster than linear interpolator
        interpolator = NearestNDInterpolator(
            self.df[['logAge', 'mass', 'Z']].values,
            self.df[['logL', 'logT', 'logR']].values,
            rescale=True,
        )
        return interpolator

    def query(self, logAge, mass, Z):
        """Query the interpolator for logL, logT, logR, and logg"""
        # Check if the input is a single value or an array
        logAge = np.atleast_1d(logAge)
        mass = np.atleast_1d(mass)
        Z = np.atleast_1d(Z)
        data_input = np.vstack([logAge, mass, Z]).T
        # Query the interpolator
        logL, logT, logR = self.interpolator(data_input).T
        # Compute logg
        LOGG_SOLAR = 4.438
        logg = LOGG_SOLAR + np.log10(mass) - 2. * logR
        data = {
            'logL': logL,
            'logT': logT,
            'logR': logR,
            'logg': logg,
        }
        return data

    def transform(self, data: Star) -> Star:
        logAge = data.logAge
        mass = data.mass
        Z = data.Z
        # Query the interpolator
        data_interp = self.query(logAge, mass, Z)
        # Update the data with the interpolated values
        data.logL = data_interp['logL']
        data.logT = data_interp['logT']
        data.logR = data_interp['logR']
        data.logg = data_interp['logg']
        return data


