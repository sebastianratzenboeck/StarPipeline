import numpy as np
from typing import List
from pyphot import sandbox as pyphot
from pyphot import svo, unit
from astropy import units as u
from base import PipelineStep
from data import Star


class Photometry(PipelineStep):
    def __init__(self, filter_names: List[str], download_from_svo=True):
        self._filter_names = None
        self._filter_bands = None
        self.download_from_svo = download_from_svo
        # Set filters
        self.set_filters(filter_names)
        self.__use_AB = ['GALEX', 'SDSS', 'PS1']

    def __str__(self):
        for pb in self._filter_bands:
            print(pb.info())

    @property
    def filter_names(self):
        return self._filter_names

    @filter_names.setter
    def filter_names(self, new_filter_names):
        self.set_filters(new_filter_names)

    def get_zero_point_flux(self, phot_filter):
        # Compute the zero points
        if phot_filter.name.split('_')[0] in self.__use_AB:
            return phot_filter.AB_zero_flux
        else:
            return phot_filter.Vega_zero_flux

    def set_filters(self, new_filter_names: List[str]):
        """Set the filters"""
        self._filter_names = new_filter_names
        pbands = []
        phot_lib = pyphot.get_library()
        for filter_name in self._filter_names:
            filter_list = phot_lib.find(filter_name)
            if len(filter_list) == 0:
                if not self.download_from_svo:
                    warn_msg = f"Filter {filter_name} not found"
                    warn_msg += " in pyphot and download_from_svo is False."
                    warn_msg += " Set download_from_svo to True to fetch it from SVO."
                    warn_msg += " This filter is not used in this simulation!"
                    raise Warning(warn_msg)
                else:
                    # Trying to fetch the filter from SVO
                    # print(f'Filter not found in pyphot, trying SVO for {filter_name}')
                    pbands += [svo.get_pyphot_filter(filter_name)]
            else:
                pbands += [phot_lib[fn] for fn in filter_list]
        self._filter_bands = pbands
        # Set the zero points
        return self

    def transform(self, data: Star) -> Star:
        """Apply the filters to the spectrum"""
        # Fetch the parameters
        flux = data.flam_dusty
        wave = data.wavelength
        # Compute the magnitudes
        flux_bands = dict()
        mag_band = dict()
        cl_band = dict()
        transmission_band = dict()
        for pb in self._filter_bands:
            lam_filter_aa = pb.wavelength.to('AA').value
            transmit_filter = pb.transmit
            # Create the filtered flux & convert to same units (just in case)
            flux_in_band = pb.get_flux(wave.value, flux.value).to(
                unit['erg'] / (unit['s'] * unit['Angstrom'] * unit['cm']**2)
            )
            # Get the zero point flux & convert to same units (just in case)
            f_zpt = self.get_zero_point_flux(pb).to(unit['erg'] / (unit['s'] * unit['Angstrom'] * unit['cm']**2))
            # Compute the magnitude
            mag_in_band = -2.5 * np.log10(flux_in_band / f_zpt) * u.mag
            # Convert to astropy units object
            flux_in_band = flux_in_band.magnitude * u.erg / u.s / u.cm**2 / u.AA
            # Get central wavelength (and convert to astropy units object)
            central_wl = pb.cl.to(unit['angstrom']).magnitude * u.AA
            # Store magnitudes and fluxes
            flux_bands[pb.name] = flux_in_band
            mag_band[pb.name] = mag_in_band
            cl_band[pb.name] = central_wl
            transmission_band[pb.name] = {'lam_aa': lam_filter_aa, 'transmit': transmit_filter}
        # Add the magnitudes to the data dictionary
        data.mag_band = mag_band
        data.flam_band = flux_bands
        data.cl_band = cl_band
        data.filter_transmission = transmission_band
        return data
