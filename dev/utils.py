import numpy as np
from scipy.interpolate import interp1d


class RobustBatchLinearInterpolator:
    def __init__(self, data, x_domain=None):
        """
        Parameters:
        - data: np.ndarray of shape (N, 20, D)
        - x_domain: optional np.ndarray of shape (20,)
        """
        self.data = data  # (N, 20, D)
        self.N, self.M, self.D = data.shape

        if self.M != 20:
            raise ValueError(f"Expected 20 domain points, got {self.M}.")

        if x_domain is None:
            self.x_domain = np.linspace(0, self.M - 1, self.M)
        else:
            self.x_domain = np.asarray(x_domain)
            if self.x_domain.shape[0] != self.M:
                raise ValueError(f"x_domain must have {self.M} elements.")

    def __call__(self, x_query):
        x_query = np.asarray(x_query)
        if x_query.shape != (self.N,):
            raise ValueError(f"x_query must have shape ({self.N},), got {x_query.shape}")

        idxs = np.searchsorted(self.x_domain, x_query, side='left')
        idxs = np.clip(idxs, 1, self.M - 1)

        x0 = self.x_domain[idxs - 1]
        x1 = self.x_domain[idxs]

        y0 = self.data[np.arange(self.N), idxs - 1]
        y1 = self.data[np.arange(self.N), idxs]

        nan_mask = np.isnan(y0) | np.isnan(y1)
        y0 = np.nan_to_num(y0, nan=0.0)
        y1 = np.nan_to_num(y1, nan=0.0)

        slope = (y1 - y0) / (x1 - x0)[:, None]
        interpolated = y0 + slope * (x_query - x0)[:, None]

        interpolated[nan_mask] = np.nan

        # Fix extrapolation
        below_domain = x_query < self.x_domain[0]
        above_domain = x_query > self.x_domain[-1]

        interpolated[below_domain] = self.data[np.where(below_domain)[0], 0]
        interpolated[above_domain] = self.data[np.where(above_domain)[0], -1]

        return interpolated



class BatchInterp1d:
    def __init__(self, data, x_domain=None):
        """
        data: (N, 20, D) array
        """
        self.data = data
        self.N, self.M, self.D = data.shape

        if x_domain is None:
            self.x_domain = np.linspace(0, self.M - 1, self.M)
        else:
            self.x_domain = np.asarray(x_domain)
            if self.x_domain.shape[0] != self.M:
                raise ValueError("x_domain must match second dimension of data.")

        # Prebuild interpolators per sample
        self.interpolators = [
            interp1d(self.x_domain, self.data[i].T, bounds_error=False, fill_value='extrapolate')
            # interp1d(self.x_domain, self.data[i], axis=0, kind='linear', fill_value='extrapolate')
            for i in range(self.N)
        ]

    def query(self, x_query):
        """
        x_query: (N,) array of query points
        returns (N, D) interpolated results
        """
        x_query = np.asarray(x_query)
        if x_query.shape != (self.N,):
            raise ValueError(f"x_query must have shape ({self.N},)")

        # Interpolate each sample individually (still fast)
        results = np.stack([
            interp(x_query[i]) for i, interp in enumerate(self.interpolators)
        ], axis=0)

        return results