import numpy as np
import pandas as pd
import inspect
import extinction
from dataclasses import dataclass, fields, field
from typing import Optional, Callable
from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS
from utils import RobustBatchLinearInterpolator



class PhysBase:
    @staticmethod
    def check_units(phys_obj: Optional[u.Quantity], required_unit: u.Unit) -> None:
        """Simple unit check."""
        if phys_obj is not None:
            # Check if the unit is equivalent to the one we want
            if not phys_obj.unit.is_equivalent(required_unit):
                raise ValueError(f"Units of {phys_obj.unit} are not equivalent to {required_unit}")
            # Check if the unit is exactly the same
            if phys_obj.unit != required_unit:
                raise ValueError(
                    f"Units of {phys_obj.unit} are equivalent to {required_unit} but not the same scale!"
                )

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert all dataclass fields (arrays) into a pandas DataFrame.
        Assumes that all fields are numpy arrays or array-like.
        """
        shape2match = self.mass.shape
        data = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # remove any leading underscores from fname
            f_name_clean = f.name.lstrip('_')
            # Check if the value is a numpy array or array-like
            if isinstance(value, (np.ndarray, list)):
                # Check if the shape matches the expected shape
                if np.asarray(value).shape == shape2match:
                    data[f_name_clean] = value

            # 2. Include all @property attributes
            for name, _ in inspect.getmembers(type(self), lambda v: isinstance(v, property)):
                if name not in data:  # Avoid overwriting
                    value = getattr(self, name)
                    if isinstance(value, (np.ndarray, list)):
                        if np.asarray(value).shape == shape2match:
                            data[name] = value

        df = pd.DataFrame(data)
        return df


@dataclass
class Star(PhysBase):
    """Class to hold the physical parameters of a star."""
    mass: np.ndarray = field(default_factory=lambda: np.array([]))
    logAge: np.ndarray = field(default_factory=lambda: np.array([]))
    Z: np.ndarray = field(default_factory=lambda: np.array([]))
    logg: Optional[np.ndarray] = None   # Surface gravity
    logT: Optional[np.ndarray] = None   # Temperature
    logL: Optional[np.ndarray] = None   # Luminosity
    _logR: Optional[np.ndarray] = None   # Radius
    # Positional parameters
    _distance: Optional[u.Quantity] = None
    _skycoords: Optional[SkyCoord] = None
    # Single "boring" star without any medium, disk or envelope
    is_single_star_without_medium: Optional[np.ndarray] = None
    # Ambient medium properties
    has_ambient_medium: Optional[np.ndarray] = None
    # Disk properties
    has_disk: Optional[np.ndarray] = None
    incl: Optional[np.ndarray] = None
    logDiskMass: Optional[np.ndarray] = None
    logDiskRmax: Optional[np.ndarray] = None
    DiskBeta: Optional[np.ndarray] = None
    DiskP: Optional[np.ndarray] = None
    logDiskH100: Optional[np.ndarray] = None
    # Envelope properties
    logEnvRho0: Optional[np.ndarray] = None
    # Cavity properties
    CavityTheta0: Optional[np.ndarray] = None
    CavityPow: Optional[np.ndarray] = None
    logCavityRho0: Optional[np.ndarray] = None

    # --------------------
    # Stellar radius in log10(R_sun)
    @property
    def logR(self) -> Optional[np.ndarray]:
        """Radius in log10(R_sun)"""
        if self._logR is not None:
            return self._logR
        # If logR is not set, compute it from logL and logT
        if (self.logL is None) or (self.logT is None):
            return None
        return self.radius_from_logL_logT(self.logL, self.logT)

    @logR.setter
    def logR(self, value: Optional[np.ndarray]) -> None:
        """Set the radius in log10(R_sun)"""
        self._logR = value

    @staticmethod
    def radius_from_logL_logT(logL, logT):
        """Radius in solar radii from logL and logT (base-10)."""
        LOGT_SUN = np.log10(5772.)  # based on IAU 2015 resolution B3 T_sun ~ 5772 K
        return 0.5*(logL - 4*(logT - LOGT_SUN)) # log10(R/R_sun)

    @property
    def is_alive(self):
        """Check if the star is alive based on the mass and logAge."""
        if (self.mass is None) or (self.logAge is None):
            return None
        return np.log10(10 ** 10 * (1 / self.mass) ** 2.5) > self.logAge

    @property
    def N(self):
        """Number of stars."""
        if self.mass is None:
            return None
        return self.mass.shape[0]

    # --------------------
    # distance
    @property
    def distance(self) -> Optional[u.Quantity]:
        return self._distance

    @distance.setter
    def distance(self, value: Optional[u.Quantity]) -> None:
        if value is not None:
            self.check_units(value, u.pc)
        self._distance = value

    # --------------------
    # skycoords
    @property
    def skycoords(self) -> Optional[SkyCoord]:
        return self._skycoords

    @skycoords.setter
    def skycoords(self, coords: Optional[SkyCoord]) -> None:
        # Transform to ICRS coordinates
        c_icrs = coords.transform_to(ICRS())
        c_icrs.representation_type = 'spherical'
        skycoords_icrs = SkyCoord(c_icrs)
        # Set the coordinates
        self._skycoords = skycoords_icrs
        self.distance = coords.distance.to(u.pc)

    @property
    def has_envelope(self):
        """
        Check if the star has an envelope.
        Envelope density is zero after ~0.5 Myr (Evans et al. 2009).
        """
        if self.logEnvRho0 is None:
            return None
        return np.isfinite(self.logEnvRho0)



@dataclass
class SEDData(Star):
    # Internal (private) storage
    Av: float = 0.0
    Rv: float = 3.1
    _av_law: Optional[Callable] = np.vectorize(extinction.fitzpatrick99, signature='(n),(),(),()->(n)')
    # Spectal parameters
    _wavelength: Optional[u.Quantity] = None
    _flam: Optional[u.Quantity] = None
    # YSO specific parameters
    _wavelength_yso: Optional[u.Quantity] = None
    _flam_yso = None
    aperture_arcsec: Optional[u.Quantity] = None
    model_used = None

    @property
    def aperture(self) -> Optional[u.Quantity]:
        """Transform aperture in arcsec to AU"""
        if (self.aperture_arcsec is None) or (self.distance is None):
            return None
        self.check_units(self.aperture_arcsec, u.arcsec)
        apertures_au = self.aperture_arcsec.value * self.distance.to(u.pc).value * u.au
        return apertures_au

    @property
    def av_law(self):
        return self._av_law

    @av_law.setter
    def av_law(self, law: Callable) -> None:
        """ Set the extinction law to be used"""
        # Vectorized version allows for Av and Rv to be arrays
        self._av_law = np.vectorize(law, signature='(n),(),(),()->(n)')
        return self

    # --------------------
    # wavelength
    @property
    def wavelength(self) -> Optional[u.Quantity]:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Optional[u.Quantity]) -> None:
        if value is not None:
            self.check_units(value, u.AA)
        self._wavelength = value

    # --------------------
    # flam
    @property
    def flam(self) -> Optional[u.Quantity]:
        return self._flam

    @flam.setter
    def flam(self, value: Optional[u.Quantity]) -> None:
        if value is not None:
            self.check_units(value, u.erg / u.s / (u.cm**2) / u.AA)
        self._flam = value

    # --------------------
    @property
    def flam_dusty(self) -> Optional[u.Quantity]:
        """Apply the dust model to the spectrum."""
        if (self.wavelength is None) or (self._flam is None) or (self.Av is None) or (self.Rv is None):
            print('')
            return None
        # Compute extincted spectrum
        Dlambda = np.exp(-1 * self._av_law(self._wavelength.value, self.Av, self.Rv, unit='aa'))
        if len(Dlambda[None, :].shape) == 1:
            # In case Av and Rv are scalars
            Dlambda = Dlambda[:, None]
        # extingted spectra
        flam_dusty = self._flam * Dlambda
        return flam_dusty

    # --------- YSO specific parameters -----------------
    # wavelength
    @property
    def wavelength_yso(self) -> Optional[u.Quantity]:
        return self._wavelength_yso

    @wavelength_yso.setter
    def wavelength_yso(self, value: Optional[u.Quantity]) -> None:
        if value is not None:
            self.check_units(value, u.AA)
        self._wavelength_yso = value

    # --------------------
    # flam
    @property
    def flam_yso(self) -> Optional[u.Quantity]:
        """Get the YSO spectrum - depends on the aperture."""
        if self.aperture is None:
            return None
        ap = self.aperture.to(u.au).value
        return self._flam_yso(ap) * u.erg / u.s / u.cm**2 / u.AA

    @flam_yso.setter
    def flam_yso(self, value: RobustBatchLinearInterpolator) -> None:
        self._flam_yso = value

    @property
    def flam_dusty_yso(self) -> Optional[u.Quantity]:
        """Apply the dust model to the spectrum."""
        if (self.wavelength_yso is None) or (self.flam_yso is None) or (self.Av is None) or (self.Rv is None):
            return None
        # Compute extincted spectrum
        Dlambda = np.exp(-1 * self._av_law(self.wavelength_yso, self.Av, self.Rv, unit='aa'))
        if len(Dlambda[None, :].shape) == 1:
            # In case Av and Rv are scalars
            Dlambda = Dlambda[:, None]
        # extingted spectra
        flam_dusty_yso = self.flam_yso * Dlambda
        return flam_dusty_yso



@dataclass
class PhotData(SEDData):
    # Photometric parameters
    _flam_band: dict = None  # Flux in the band
    _mag_band: dict = None   # Magnitude in the band
    _cl_band: dict = None    # Central wavelength

    # --------------------
    # flam_band
    @property
    def flam_band(self) -> dict:
        return self._flam_band

    @flam_band.setter
    def flam_band(self, value: dict) -> None:
        if value is not None:
            for band_name, flam_in_band in value.items():
                # Check if the band is a valid filter
                self.check_units(flam_in_band, u.erg / u.s / (u.cm**2) / u.AA)
        self._flam_band = value

    # --------------------
    # mag_band
    @property
    def mag_band(self) -> dict:
        return self._mag_band

    @mag_band.setter
    def mag_band(self, value: dict) -> None:
        if value is not None:
            for band_name, mag_in_band in value.items():
                self.check_units(mag_in_band, u.mag)
        self._mag_band = value

    # --------------------
    # cl_band
    @property
    def cl_band(self) -> dict:
        return self._cl_band

    @cl_band.setter
    def cl_band(self, value: dict) -> None:
        if value is not None:
            for band_name, cl_in_band in value.items():
                self.check_units(cl_in_band, u.AA)
        self._cl_band = value

