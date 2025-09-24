import os
import numpy as np
from dustmaps.planck import PlanckQuery
from dustmaps.bayestar import BayestarQuery
from dustmaps.edenhofer2023 import Edenhofer2023Query
from base import PipelineStep
from data import SEDData


class DustMap(PipelineStep):
    def __init__(self, dustmap_name: str, map_base_path: str, **kwargs):
        self.map_base_path = map_base_path
        self._dustmap = None
        self.__map_conversion_to_Av = None
        self.__query_kwargs = {}
        # Set the dustmap
        self.set_dustmap(dustmap_name, **kwargs)

    def __str__(self):
        return f'Applying extinction map: {self._dustmap}'

    @property
    def dustmap(self):
        return self._dustmap

    @dustmap.setter
    def dustmap(self, new_dustmap):
        self.set_dustmap(new_dustmap)

    @property
    def query_kwargs(self):
        return self.__query_kwargs

    @query_kwargs.setter
    def query_kwargs(self, new_query_kwargs):
        self.__query_kwargs = new_query_kwargs

    def set_dustmap(self, new_dustmap, **kwargs):
        if new_dustmap.lower() == 'edenhofer':
            # Hard coded dustmap name for now --> add added flexibility to change the dustmap via setter function
            mode = kwargs.pop('mode', 'mean')   # mean, samples, or random_sample
            if mode == 'mean':
                fname_map = os.path.join(self.map_base_path, 'mean_and_std_healpix.fits')
            else:
                fname_map = os.path.join(self.map_base_path, 'samples_healpix.fits')
                mode = 'random_sample'
            # Extract query kwargs
            self.__query_kwargs = dict(mode=mode)
            # Set the dustmap
            load_samples = kwargs.pop('load_samples', False)
            flavor = kwargs.pop('flavor', 'main')
            self._dustmap = Edenhofer2023Query(
                map_fname=fname_map,
                integrated=True,
                load_samples=load_samples,
                flavor=flavor,
                **kwargs
            )
            self.__map_conversion_to_Av = lambda map_values, Rv=None: map_values * 2.8
        elif new_dustmap.lower() == 'planck':
            # Dustmap path
            fname_map = os.path.join(self.map_base_path, 'HFI_CompMap_ThermalDustModel_2048_R1.20.fits')
            # Extract query kwargs
            self.__query_kwargs = dict()
            self._dustmap = PlanckQuery(map_fname=fname_map, **kwargs)
            self.__map_conversion_to_Av = lambda map_values, Rv: Rv * map_values
        elif new_dustmap.lower() == 'bayestar':
            # Dustmap path
            fname_map = os.path.join(self.map_base_path, 'bayestar2019.h5')
            # Extract query kwargs
            mode = kwargs.pop('mode', 'random_sample')
            return_flags = kwargs.pop('return_flags', False)
            pct = kwargs.pop('pct', None)
            self.__query_kwargs = dict(
                mode=mode,
                return_flags=return_flags,
                pct=pct
            )
            # Set the dustmap
            kwargs.pop('version', None)  # remove version from kwargs, currently only bayestar2019 is available
            self._dustmap = BayestarQuery(
                map_fname=fname_map,
                version='bayestar2019',
                **kwargs
            )
            self.__map_conversion_to_Av = lambda map_values, Rv=None: 2.742 * map_values
        else:
            msg = f'"{new_dustmap}" is not a valid dustmap name.'
            msg += ' Valid options are: `edenhofer`, `planck`, or `bayestar`.'
            raise ValueError(msg)

    def transform(self, data: SEDData) -> SEDData:
        """Generate spectra for the given parameters"""
        # Fetch the parameters
        skycoords = data.skycoords
        Rv = data.Rv
        # Compute the spectra
        if skycoords is not None:
            map_values = self._dustmap.query(skycoords, **self.__query_kwargs)
            A_V = self.__map_conversion_to_Av(map_values, Rv)
            # Convert nan values to zero
            A_V[np.isnan(A_V)] = 0
            data.Av = A_V
        return data
