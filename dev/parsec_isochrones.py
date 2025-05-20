import re
import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from base import PipelineStep
from data import Star


class ICBase:
    def __init__(self):
        self.data = None
        self.colnames = None
        self.cols_input = None
        self.cols_predict = None
        self.predict_clean_names = None
        self.l_interp = None

    def fit_interpolator(self, n_skip=10):
        df_subset = self.data[::n_skip]
        # Only model mass and age for now
        X = df_subset[self.cols_input].values
        y = df_subset[self.cols_predict].values
        self.l_interp = LinearNDInterpolator(X, y)
        # self.l_interp = NearestNDInterpolator(X, y)

    def query_cmd(self, mass, logage, feh):
        if not isinstance(mass, np.ndarray):
            mass = np.array([mass])
            logage = np.array([logage])
            feh = np.array([feh])
        # Query the interpolator
        X_query = np.vstack([mass, logage, feh]).T
        # Interpolate
        df = pd.DataFrame(
            self.l_interp(X_query),
            columns=self.predict_clean_names
        )
        return df

    def query_cmd_Z(self, mass, logage, Z):
        feh = self.feh_from_z(Z)
        return self.query_cmd(mass, logage, feh)

    @staticmethod
    def feh_from_z(Z, z_x_sun=0.0207, y_primordial=0.2485, dY_dZ=1.78):
        """
        Convert total metallicity Z to [Fe/H] using:
        - Z/X_sun = 0.0207
        - Y = Y_p + (dY/dZ) * Z
        - X = 1 - Y - Z
        """
        Z = np.asarray(Z)  # Ensure input is array-like
        Y = y_primordial + dY_dZ * Z
        X = 1 - Y - Z
        Z_X = Z / X
        feh = np.log10(Z_X / z_x_sun)
        return feh


class ParsecBase(ICBase):
    """Handling PARSEC isochrones"""
    def __init__(self, dir_path, file_ending='dat'):
        super().__init__()
        # Save some PARSEC internal column names
        self.comment = r'#'
        self.colnames = {'header_start': '# Zini', 'logTeff': 'logTe'}
        # self.post_process = {self.colnames['teff']: lambda x: 10 ** x}
        self.post_process = {}
        self.dir_path = dir_path
        self.flist_all = glob.glob(os.path.join(dir_path, f'*.{file_ending}'))

    def read_files(self, flist):
        frames = []
        for fname in flist:
            df_iso = self.read(fname)
            # Postprocessing
            for col, func in self.post_process.items():
                df_iso[col] = df_iso[col].apply(func)
            frames.append(df_iso)
        print('PARSEC isochrones read and processed!')
        df_final = pd.concat(frames)
        # Remove labels 8 & 9
        df_final = df_final[~df_final['label'].isin([7, 8, 9])]
        return df_final.sort_values(by=[self.colnames['logAge'], self.colnames['Z'], self.colnames['mass']])

    def read(self, fname):
        """
        Fetches the coordinates of a single isochrone from a given file and retrurns it
        :param fname: File name containing information on a single isochrone
        :return: x-y-Coordinates of a chosen isochrone, age, metallicity
        """
        # delim_whitespace is deprecated --> use sep='\s+' instead
        # df_iso = pd.read_csv(fname, delim_whitespace=True, comment=self.comment, header=None)
        df_iso = pd.read_csv(fname, sep='\s+', comment=self.comment, header=None)
        # Read first line and extract column names
        with open(fname) as f:
            for line in f:
                if line.startswith(self.colnames['header_start']):
                    break
        line = re.sub(r'\t', ' ', line)  # remove tabs
        line = re.sub(r'\n', ' ', line)  # remove newline at the end
        line = re.sub(self.comment, ' ', line)  # remove '#' at the beginning
        col_names = [elem for elem in line.split(' ') if elem != '']
        # Add column names
        df_iso.columns = col_names
        return df_iso


class ParsecInterpolator(ParsecBase):
    """Handling parsec isochrones"""
    def __init__(self, dir_path, file_ending='dat'):
        super().__init__(dir_path, file_ending)
        # Save some PARSEC internal column names
        self.colnames = {
            'Z': 'Zini',
            'mass': 'Mini',
            'logg': 'logg',
            'logT': 'logTe',
            'logL': 'logL',
            'logAge': 'logAge',
            'feh': 'MH',
            'header_start': '# Zini'
        }
        # Save data and rename columns
        self.data = self.read_files(self.flist_all)
        # Prepare interpolation method
        self.cols_input = [self.colnames['mass'], self.colnames['logAge'], self.colnames['feh']]
        self.cols_predict = [
            self.colnames['logg'], self.colnames['logT'], self.colnames['logL'],
        ]
        self.predict_clean_names = ['logg', 'logT', 'logL']
        self.fit_interpolator(n_skip=5)


class Parsec(ParsecInterpolator, PipelineStep):
    def __init__(self, dir_path, file_ending='dat'):
        super().__init__(dir_path, file_ending)

    def __str__(self):
        return f"ParsecInterpolator: using following files {self.flist_all}"

    def transform(self, data: Star) -> Star:
        """Get physical stellar paramters from masses, logAge, and metallicity"""
        # Fetch the parameters
        mass = data.mass
        logAge = data.logAge
        Z = data.Z
        # Compute the stellar parameters
        logg, logT, logL = self.query_cmd_Z(mass, logAge, Z).values.T
        # Update the data dictionary with the new parameters
        data.logg = logg
        data.logT = logT
        data.logL = logL
        return data
