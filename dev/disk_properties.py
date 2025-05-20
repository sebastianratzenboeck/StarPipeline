import numpy as np
from base import PipelineStep
from data import Star


class CondDiskProperties(PipelineStep):
    def __init__(self, disk_decay_myr=6.0, ambient_medium_decay=4.0):
        self.t_disk_decay_myr = disk_decay_myr  # Myr
        self.t_ambient_medium_decay = ambient_medium_decay  # Myr

    def has_disk(self, logAge):
        """
        Calculate the disk fraction based on the logAge.
        """
        t_myr = 10 ** logAge / 1e6  # Convert to Myr
        # Disk fraction decreases exponentially with time
        disk_fraction = np.exp(-t_myr / self.t_disk_decay_myr)
        has_disk = np.random.rand(t_myr.size) < disk_fraction
        return has_disk

    def has_ambient_medium(self, logAge):
        """
        Calculate the ambient medium fraction based on the logAge.
        """
        t_myr = 10 ** logAge / 1e6
        # Ambient medium fraction decreases exponentially with time
        ambient_fraction = np.exp(-t_myr / self.t_ambient_medium_decay)
        has_ambient = np.random.rand(t_myr.size) < ambient_fraction
        return has_ambient

    def sample_disk_mass(self, logAge, mass):
        """
        Disk mass scales roughly linearly with stellar mass and decreases with age (Richardson et al. 2024).
        """
        t_myr = 10 ** logAge  # Convert to yr
        mu_log10_disk_mass = np.log10(0.01 * mass) - np.log10(t_myr / 1e5)
        # median_log_disk_mass = np.log10(0.01 * self.mass) - (10**self.logAge / 5.0)
        sigma_log_disk_mass = 0.5
        disk_mass = np.random.normal(mu_log10_disk_mass, sigma_log_disk_mass)
        return np.clip(disk_mass, -8, -1)

    def sample_disk_rmax(self, logAge, mass):
        """
        Disk outer radius increases as the system evolves and depends weakly on stellar mass
        (Robitaille et al. 2006).
        """
        age_myr = 10 ** logAge / 1e6
        min_rmax = 10 + 20 * age_myr
        max_rmax = 100 + 200 * age_myr
        rmax = 10 ** np.random.uniform(np.log10(min_rmax), np.log10(max_rmax))
        rmax *= mass ** 0.2
        return np.clip(np.log10(rmax), np.log10(50), np.log10(5000))

    def sample_disk_beta(self, logAge):
        """
        Disk flaring index generally ranges from 1.0 to 1.3; younger disks are more flared (Richardson et al. 2017).
        """
        # ---- old way ----
        # t_star = 10**self.logAge  # Convert to yr
        # beta_mu = 1.2 - 0.05 * np.log10(t_star/1e5)
        # disk_beta = np.random.normal(beta_mu, 0.05)
        # ---- new way ----
        # Base beta value
        N = logAge.shape[0]
        base_beta = np.random.uniform(1.0, 1.3, N)
        age_myr = 10 ** logAge / 1e6
        base_beta[age_myr < 0.5] += 0.05
        return np.clip(base_beta, 1.0, 1.3)

    def sample_disk_p(self, N):
        """
        Disk surface density power-law index typically between -2 and 0 (Richardson et al. 2017).
        """
        return np.random.uniform(-2, 0, N)

    def sample_disk_h100(self, logL):
        """
        Disk scale height at 100 AU depends slightly on stellar luminosity (Robitaille et al. 2017).
        """
        N = logL.shape[0]
        SCALE_H100_MIN_LOG = np.log10(1)
        SCALE_H100_MAX_LOG = np.log10(20)
        # Sample a uniform distribution between the min and max scale height
        log_h100 = np.random.uniform(SCALE_H100_MIN_LOG, SCALE_H100_MAX_LOG, N)
        log_h100 += 0.1 * logL
        return np.clip(log_h100, SCALE_H100_MIN_LOG, SCALE_H100_MAX_LOG)

    def sample_envelope_density(self, logAge):
        """
        Envelope density decreases as the system evolves, effectively zero after ~1 Myr
        (Evans et al. 2009; Young & Evans 2005).
        Envelope function is
            i) approx. constant for t < 10^4 yr (Shu 1977; Terebey et al. 1984)
            ii) decreases between 10^5 and 10^6 yr (Foster & Chevalier 1993; Foster 1994; Hartmann 2001, p. 237),
            iii) effectively zero after 0.5-1 Myr (Evans et al. 2009; Young & Evans 2005).
        """
        age_myr = 10 ** logAge / 1e6
        rho_env = np.full_like(age_myr, fill_value=-np.inf)
        # has envelope -> make exponential decay
        exp_dec_env = np.exp(-age_myr / 1.)
        has_env = np.random.rand(age_myr.size) < exp_dec_env
        if has_env.any():
            mean_log_rho = -18 - (age_myr[has_env] * 4)
            rho_env[has_env] = np.random.normal(mean_log_rho, 1)
            rho_env[has_env] = np.clip(rho_env[has_env], -24, -16)
        return rho_env

    def sample_cavity_theta0(self, logAge):
        """
        Cavity half-opening angle widens significantly with age due to outflows (Arce & Sargent 2006).
        """
        age_myr = 10 ** logAge / 1e6
        min_theta = 5 + 80 * (age_myr / 0.5)
        max_theta = 10 + 100 * (age_myr / 0.5)
        theta0 = np.random.uniform(min_theta, max_theta)
        return np.clip(theta0, 0, 60)

    def sample_cavity_power(self, theta0):
        """
        Wider cavities typically exhibit parabolic shapes, thus higher power index (Offner et al. 2022).
        """
        cavity_power = 1 + (theta0 / 60)
        return np.clip(cavity_power, 1.0, 2.0)

    def sample_cavity_density(self, N):
        """
        Cavity density typically much lower than envelope density, ranging from 1e-23 to 1e-19 g/cm^3
        (Robitaille et al. 2017).
        """
        return np.random.uniform(-23, -19, N)

    def sample_inclination(self, N):
        """
        Viewing inclination is random and isotropic, uniform in cos(i).
        """
        return np.degrees(np.arccos(np.random.uniform(0, 1, N)))

    def transform(self, data: Star) -> Star:
        """ Sample and return all parameters. """
        logAge = data.logAge
        mass = data.mass
        logL = data.logL
        N = logAge.shape[0]

        # Determine if the star has an ambient medium
        has_ambient_medium = self.has_ambient_medium(logAge)
        # Sample disk properties
        has_disk = self.has_disk(logAge)
        disk_mass = np.where(has_disk, self.sample_disk_mass(logAge, mass), np.nan)
        disk_rmax = np.where(has_disk, self.sample_disk_rmax(logAge, mass), np.nan)
        disk_beta = np.where(has_disk, self.sample_disk_beta(logAge), np.nan)
        disk_p = np.where(has_disk, self.sample_disk_p(N), np.nan)
        disk_h100 = np.where(has_disk, self.sample_disk_h100(logL), np.nan)
        inclination = np.where(has_disk, self.sample_inclination(N), np.nan)
        # Envelope and cavity properties
        envelope_rho = self.sample_envelope_density(logAge)
        has_envelope = np.isfinite(envelope_rho)
        cavity_theta0 = np.where(has_envelope, self.sample_cavity_theta0(logAge), np.nan)
        cavity_power = np.where(has_envelope, self.sample_cavity_power(cavity_theta0), np.nan)
        cavity_rho = np.where(has_envelope, self.sample_cavity_density(N), np.nan)

        # Set the properties in the data object
        data.has_ambient_medium = has_ambient_medium
        data.has_disk = has_disk
        data.incl = inclination
        data.logDiskMass = disk_mass
        data.logDiskRmax = disk_rmax
        data.DiskBeta = disk_beta
        data.DiskP = disk_p
        data.logDiskH100 = disk_h100
        data.logEnvRho0 = envelope_rho
        data.CavityTheta0 = cavity_theta0
        data.CavityPow = cavity_power
        data.logCavityRho0 = cavity_rho
        return data