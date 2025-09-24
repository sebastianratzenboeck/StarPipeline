# Create a representative grid in logAge, Z, and, mass
import numpy as np
from dataclasses import dataclass

Z_SUN = 0.0152  # Value from PARSEC

@dataclass
class ParamGrid:
    logAge: np.ndarray     # (N,) log10(age/yr)
    Z: np.ndarray          # (N,)
    M: np.ndarray          # (N,) Msun
    mask_ms_ok: np.ndarray # (N,) True if age <= t_MS(M)
    FeH: np.ndarray        # (N,) the [Fe/H] used

# ---------- helpers ----------

def lhs(n_samples: int, dim: int, seed: int = 0) -> np.ndarray:
    """Simple Latin Hypercube in [0,1]^dim."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n_samples)
        X[:, j] = (perm + rng.random(n_samples)) / n_samples
    return X

def ms_lifetime_yrs(M_msun: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Very rough MS lifetime scaling (good enough to mask impossible combos).
    t_MS ~ 1e10 yr * (M/Msun)^-2.5 * (Z/Zsun)^{+0.1}  (weak Z dependence).
    """
    M = np.asarray(M_msun, float)
    Zratio = np.clip(Z / Z_SUN, 1e-3, 1e3)
    return 1e10 * (M ** -2.5) * (Zratio ** 0.1)

# ---------- Latin Hypercube grid (recommended for “representative” coverage) ----------

def make_representative_grid_lhs(
    n_samples: int = 5000,
    logAge_range = (6.0, 10.0),
    Z_range      = (0.005, 0.03),
    M_range      = (0.08, 8.0),
    global_Z_floor = 1e-4,     # clip to broad physical/model limits if needed
    global_Z_ceil  = 0.06,
    enforce_ms: bool = True,
    seed: int = 42,
) -> ParamGrid:
    """
    Returns N samples covering the domain with realistic spacing:
      - logAge uniform in [6,10]
      - metallicity uniform in [Fe/H] then mapped to Z in the requested Z_range
      - mass uniform in log10 M
      - optional main-sequence lifetime mask
    """
    # Map the requested Z-range into [Fe/H] bounds
    Z_lo = np.clip(Z_range[0], global_Z_floor, global_Z_ceil)
    Z_hi = np.clip(Z_range[1], global_Z_floor, global_Z_ceil)
    FeH_lo = np.log10(Z_lo / Z_SUN)
    FeH_hi = np.log10(Z_hi / Z_SUN)

    U = lhs(n_samples, dim=3, seed=seed)  # columns: [u_age, u_FeH, u_mass]

    # logAge
    logAge = logAge_range[0] + U[:, 0] * (logAge_range[1] - logAge_range[0])

    # metallicity via uniform [Fe/H]
    FeH = FeH_lo + U[:, 1] * (FeH_hi - FeH_lo)
    Z   = Z_SUN * (10.0 ** FeH)
    Z   = np.clip(Z, global_Z_floor, global_Z_ceil)

    # mass via uniform in log10 M
    logM_lo, logM_hi = np.log10(M_range[0]), np.log10(M_range[1])
    logM = logM_lo + U[:, 2] * (logM_hi - logM_lo)
    M = 10.0 ** logM

    # (optional) mask combos that cannot be on MS at that age
    if enforce_ms:
        t_ms = ms_lifetime_yrs(M, Z)
        mask = (10.0 ** logAge) <= t_ms
    else:
        mask = np.ones(n_samples, dtype=bool)

    return ParamGrid(logAge=logAge, Z=Z, M=M, mask_ms_ok=mask, FeH=FeH)

# ---------- Tensor (Cartesian) grid variant ----------

def make_tensor_grid(
    n_age: int = 32, n_Z: int = 16, n_M: int = 32,
    logAge_range = (6.0, 10.0),
    Z_range      = (0.005, 0.03),
    M_range      = (0.08, 8.0),
    global_Z_floor = 1e-4, global_Z_ceil = 0.06,
    enforce_ms: bool = True,
):
    """
    Returns a ParamGrid with flattened arrays from a Cartesian product.
    Spacing:
      - logAge: linspace in [6,10]
      - [Fe/H]: linspace, then map to Z
      - mass: logspace in [0.08, 8]
    """
    # logAge
    logAge = np.linspace(logAge_range[0], logAge_range[1], n_age)

    # metallicity via uniform [Fe/H]
    Z_lo = np.clip(Z_range[0], global_Z_floor, global_Z_ceil)
    Z_hi = np.clip(Z_range[1], global_Z_floor, global_Z_ceil)
    FeH_lo = np.log10(Z_lo / Z_SUN)
    FeH_hi = np.log10(Z_hi / Z_SUN)
    FeH = np.linspace(FeH_lo, FeH_hi, n_Z)
    Z = Z_SUN * (10.0 ** FeH)

    # mass (log grid)
    M = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), n_M)

    LA, ZZ, MM = np.meshgrid(logAge, Z, M, indexing='ij')
    la = LA.ravel(); z = ZZ.ravel(); mm = MM.ravel()
    feh = np.log10(z / Z_SUN)

    if enforce_ms:
        mask = (10.0 ** la) <= ms_lifetime_yrs(mm, z)
    else:
        mask = np.ones_like(la, dtype=bool)

    return ParamGrid(logAge=la, Z=z, M=mm, mask_ms_ok=mask, FeH=feh)