import imf
import numpy as np
import pandas as pd
from base import PipelineStep
from astropy import units as u
from astropy.coordinates import ICRS, Galactic, SkyCoord
from data import Star


class StarCluster(PipelineStep):
    def __init__(self, mu: np.ndarray, cov: np.ndarray, cluster_mass: float, logAge: float, Z: float):
        # Cluster geometry
        self.mu = mu
        self.cov = cov
        self.X_gal = None
        # physical parameters
        self.cluster_mass = cluster_mass
        self.logAge = logAge
        self.Z = Z
        # Computed when simulated
        self.mass_samples = None
        self.lifetime_logAge = None

    def simulate_cluster(self):
        self.mass_samples = np.sort(imf.make_cluster(self.cluster_mass))
        self.lifetime_logAge = np.log10(10 ** 10 * (1 / self.mass_samples) ** 2.5)
        cluster_size = len(self.mass_samples)
        # Draw samples from mv normal
        self.X_gal = pd.DataFrame(
            np.random.multivariate_normal(self.mu, self.cov, cluster_size),
            columns=['X', 'Y', 'Z', 'U', 'V', 'W']
        )
        return self

    @staticmethod
    def skycoord_from_galactic(data_gal):
        X, Y, Z, U, V, W = data_gal.T
        c = Galactic(
            u=X * u.pc, v=Y * u.pc, w=Z * u.pc,
            U=U * u.km / u.s, V=V * u.km / u.s, W=W * u.km / u.s,
            representation_type="cartesian",
            # Velocity representation
            differential_type="cartesian",
        )
        c_icrs = c.transform_to(ICRS())
        c_icrs.representation_type = 'spherical'
        skycoords = SkyCoord(c_icrs)
        return skycoords

    def transform(self, data: Star) -> Star:
        # Simulate cluster
        self.simulate_cluster()
        skycoords = self.skycoord_from_galactic(self.X_gal.values)
        # Save masses, logAge, etc.
        logAge_samples = np.full_like(self.mass_samples, self.logAge)
        Z_samples = np.full_like(self.mass_samples, self.Z)
        # Save important stuff
        data.logAge = logAge_samples
        data.Z = Z_samples
        data.mass = self.mass_samples
        data.skycoords = skycoords
        return data
