"""
cwcgrid.py

Lightweight query interface for the CWC synthetic spectral grid.

Provides:
- Grid introspection (parameter names, ranges, counts, etc.).
- Convenient selection by stellar parameters.
- Access to fluxed and normalized spectra with optional wavelength subsetting.

Assumed default HDF5 layout
---------------------------
Top-level datasets:
    /wavelengths      : float64, (N_lambda,)
    /spectra          : float32/float64, (N_model, N_lambda)  # fluxed
    /spectra_norm     : float32/float64, (N_model, N_lambda)  # normalized (optional)
    /labels           : structured array, (N_model,) with fields like:
                        'logt', 'logg', 'feh', 'afe', 'vmic', ...

If your file uses different dataset names, you can override them when
constructing CWCFile and/or CWCGrid.
"""

from __future__ import annotations

import os, glob
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

# ---------------------------------------------------------------------
# Small container for grid metadata
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GridInfo:
    """Summary metadata describing a CWC grid file."""
    path: str
    n_models: int
    n_lambda: int
    param_names: Tuple[str, ...]
    lambda_min: float
    lambda_max: float
    available_spectra: Tuple[str, ...]  # e.g. ("flux", "norm")


# ---------------------------------------------------------------------
# Per-file interface
# ---------------------------------------------------------------------

class CWCFile:
    """
    High-level interface to a single CWC HDF5 spectral grid file.

    Example
    -------
    >>> from cwcgrid import CWCFile
    >>> spec = CWCFile("cwc_v1.0.h5")
    >>> info = spec.info
    >>> print(info.n_models, info.param_names)

    >>> # Inspect parameter ranges
    >>> spec.param_limits("logt")
    (3.4, 4.6)

    >>> # Get nearest model to a point in label-space
    >>> lam, f = spec.get_model(logt=3.76, logg=4.40, feh=0.0, afe=0.0,
    ...                         vmic=1.0, kind="flux")

    >>> # Select all models in a box in parameter space
    >>> idx, pars = spec.select(logt=(3.7, 3.8), feh=(-0.5, 0.5))
    >>> lam, specs = spec.get_spectra(idx, kind="norm", wave_range=(3800., 5200.))
    """

    def __init__(
        self,
        path: str,
        *,
        wave_key: str = "wavelengths",
        labels_key: str = "parameters",
        flux_key: str = "spectra",
        norm_key: Optional[str] = "norm",
        mode: str = "r",
    ) -> None:
    
        """
        Parameters
        ----------
        path : str
            Path to the CWC HDF5 file.
        wave_key : str, optional
            Dataset name for the wavelength axis.
        labels_key : str, optional
            Dataset name for the structured table of model parameters.
        flux_key : str, optional
            Dataset name for fluxed spectra.
        norm_key : str or None, optional
            Dataset name for normalized spectra. If None or not found,
            normalized spectra will be unavailable.
        mode : {"r", "r+"}, optional
            File open mode (usually "r" for read-only).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self._path = str(path)
        self._file = h5py.File(path, mode)

        # ---- wavelength axis ----
        if wave_key not in self._file:
            raise KeyError(f"Could not find wavelength dataset '{wave_key}' in {path}")
        self._wave_key = wave_key
        self._wave = np.array(self._file[wave_key], dtype=np.float64)
        if self._wave.ndim != 1:
            raise ValueError(f"'{wave_key}' must be 1D (got shape {self._wave.shape})")

        # ---- parameter table ----
        if labels_key not in self._file:
            raise KeyError(f"Could not find labels dataset '{labels_key}' in {path}")
        self._labels_key = labels_key
        self._labels = self._file[labels_key]  # h5py Dataset (structured)
        if self._labels.ndim != 1:
            raise ValueError(f"'{labels_key}' must be 1D structured array")

        # Ensure we have named fields
        if self._labels.dtype.names is None:
            raise ValueError(f"'{labels_key}' must be a structured array with named fields")

        # ---- spectra datasets ----
        if flux_key not in self._file:
            raise KeyError(f"Could not find fluxed spectra dataset '{flux_key}' in {path}")
        self._flux_key = flux_key
        self._flux = self._file[flux_key]

        if self._flux.ndim != 2:
            raise ValueError(f"'{flux_key}' must have shape (N_model, N_lambda)")

        # Optional normalized spectra (if norm_key points to a *dataset*).
        # In your current CWC files, 'norm' is a GROUP of stats, so this
        # will detect that and ignore it for spectra access.
        self._norm_key = None
        self._norm = None
        if norm_key is not None and norm_key in self._file:
            obj = self._file[norm_key]
            if isinstance(obj, h5py.Dataset):
                self._norm_key = norm_key
                self._norm = obj
                if self._norm.shape != self._flux.shape:
                    raise ValueError(
                        f"'{norm_key}' must have same shape as '{flux_key}' "
                        f"(got {self._norm.shape} vs {self._flux.shape})"
                    )
            # else: it's a Group (like your current 'norm'); ignore for spectra

        # Optional continuum spectra: CWC uses 'continuua'
        self._cont_key = None
        self._cont = None
        if "continuua" in self._file:
            cont = self._file["continuua"]
            if isinstance(cont, h5py.Dataset):
                self._cont_key = "continuua"
                self._cont = cont
                if self._cont.shape != self._flux.shape:
                    raise ValueError(
                        f"'continuua' must have same shape as '{flux_key}' "
                        f"(got {self._cont.shape} vs {self._flux.shape})"
                    )

        # Basic consistency checks
        if self._flux.shape[1] != self._wave.size:
            raise ValueError(
                f"Spectra wavelength dimension ({self._flux.shape[1]}) "
                f"doesn't match wavelengths axis ({self._wave.size})"
            )
        if self._flux.shape[0] != self._labels.shape[0]:
            raise ValueError(
                f"Number of models in spectra ({self._flux.shape[0]}) "
                f"doesn't match labels ({self._labels.shape[0]})"
            )

        self._n_models, self._n_lambda = self._flux.shape
        self._param_names = self._labels.dtype.names

        available_spectra = ["flux"]
        if self._norm is not None:
            available_spectra.append("norm")

        self._info = GridInfo(
            path=self._path,
            n_models=self._n_models,
            n_lambda=self._n_lambda,
            param_names=tuple(self._param_names),
            lambda_min=float(self._wave[0]),
            lambda_max=float(self._wave[-1]),
            available_spectra=tuple(available_spectra),
        )

    # -----------------------------------------------------------------
    # Context manager & cleanup
    # -----------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying HDF5 file."""
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None  # type: ignore[attr-defined]

    def __enter__(self) -> "CWCFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; safe even if already closed.
        try:
            self.close()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Basic properties
    # -----------------------------------------------------------------
    @property
    def info(self) -> GridInfo:
        """Return a lightweight summary of the grid file."""
        return self._info

    @property
    def wavelengths(self) -> np.ndarray:
        """1D wavelength axis in Ångström (copy)."""
        return self._wave.copy()

    @property
    def param_names(self) -> Tuple[str, ...]:
        """Tuple of parameter names available in the grid."""
        return self._param_names

    @property
    def n_models(self) -> int:
        return self._n_models

    @property
    def n_lambda(self) -> int:
        return self._n_lambda

    # -----------------------------------------------------------------
    # Parameter table helpers
    # -----------------------------------------------------------------
    def get_param_table(self, copy: bool = True) -> np.ndarray:
        """
        Return the full parameter table as a NumPy structured array.

        Parameters
        ----------
        copy : bool, optional
            Currently both branches return a NumPy array in memory. This
            argument is kept for potential future optimization.
        """
        if copy:
            return np.array(self._labels)
        else:
            return self._labels[...]

    def param_limits(self, name: str) -> Tuple[float, float]:
        """
        Return (min, max) for a parameter.

        Raises KeyError if the parameter is not present.
        """
        if name not in self._param_names:
            raise KeyError(f"Parameter '{name}' not found. Available: {self._param_names}")
        vals = np.array(self._labels[name], dtype=float)
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    def describe(self, print_out: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Return a summary of parameter limits and optionally print them.

        Returns
        -------
        summary : dict
            {param_name: {"min": ..., "max": ..., "mean": ..., "std": ...}, ...}
        """
        summary: Dict[str, Dict[str, float]] = {}
        for name in self._param_names:
            arr = np.array(self._labels[name], dtype=float)
            summary[name] = {
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
            }

        if print_out:
            print(f"CWCFile: {self._path}")
            print(f"  N_models  = {self._n_models}")
            print(f"  N_lambda  = {self._n_lambda}")
            print(f"  lambda    = [{self._wave[0]:.1f}, {self._wave[-1]:.1f}] Å")
            print(f"  spectra   = {self._info.available_spectra}")
            print("  parameters:")
            for name, stats in summary.items():
                print(
                    f"    {name:8s} : "
                    f"[{stats['min']:.4g}, {stats['max']:.4g}] "
                    f"(mean={stats['mean']:.4g}, std={stats['std']:.4g})"
                )

        return summary

    # -----------------------------------------------------------------
    # Parameter selection
    # -----------------------------------------------------------------
    def _build_mask(
        self,
        constraints: Mapping[str, Union[float, Tuple[float, float]]],
    ) -> np.ndarray:
        """
        Convert constraints into a boolean mask over models.

        constraints: dict(param -> value or (min, max))

        Examples
        --------
        {"feh": -0.5}             → exact match on [Fe/H] (within eps)
        {"logt": (3.7, 3.8)}      → 3.7 <= logt <= 3.8
        {"logg": 4.0, "feh": 0.0} → combined mask.
        """
        if not constraints:
            return np.ones(self._n_models, dtype=bool)

        mask = np.ones(self._n_models, dtype=bool)
        for name, val in constraints.items():
            if name not in self._param_names:
                raise KeyError(f"Parameter '{name}' not found in grid")

            arr = np.array(self._labels[name], dtype=float)

            if isinstance(val, (tuple, list)):
                if len(val) != 2:
                    raise ValueError(f"Range for '{name}' must be (min, max); got {val}")
                vmin, vmax = float(val[0]), float(val[1])
                mask &= (arr >= vmin) & (arr <= vmax)
            else:
                # Treat as an exact value with small tolerance
                target = float(val)
                tol = 1e-6  # manual tolerance; adjust if needed
                mask &= np.abs(arr - target) <= tol

        return mask

    def select(
        self,
        **constraints: Union[float, Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select models satisfying given constraints on parameters.

        Parameters
        ----------
        constraints : keyword arguments
            param=value or param=(min, max). All constraints are ANDed.

        Returns
        -------
        idx : np.ndarray
            1D array of selected model indices.
        pars : np.ndarray
            Structured array of the selected parameter rows.
        """
        mask = self._build_mask(constraints)
        idx = np.nonzero(mask)[0]
        pars = np.array(self._labels[idx])
        return idx, pars

    # -----------------------------------------------------------------
    # Spectra access
    # -----------------------------------------------------------------
    def _wave_slice(
        self,
        wave_range: Optional[Tuple[float, float]],
    ) -> Tuple[slice, np.ndarray]:
        """
        Convert a wavelength range into a slice and the corresponding λ grid.

        If wave_range is None, returns full range.
        """
        if wave_range is None:
            sl = slice(None)
            lam = self._wave
        else:
            wmin, wmax = float(wave_range[0]), float(wave_range[1])
            if wmin > wmax:
                raise ValueError("wave_range must be (min, max) with min <= max")

            # Use searchsorted assuming λ is sorted (true for CWC grid)
            i0 = int(np.searchsorted(self._wave, wmin, side="left"))
            i1 = int(np.searchsorted(self._wave, wmax, side="right"))
            i0 = max(0, min(i0, self._n_lambda))
            i1 = max(i0, min(i1, self._n_lambda))
            sl = slice(i0, i1)
            lam = self._wave[sl]

        return sl, lam

    def _get_dataset_for_kind(self, kind: str):
        """
        Map kind ('flux' or 'norm') → underlying HDF5 dataset.
        """
        kind = kind.lower()
        if kind == "flux":
            return self._flux
        elif kind == "norm":
            if self._norm is None:
                raise ValueError(
                    "Normalized spectra ('norm') requested but no "
                    f"'{self._norm_key}' dataset is present in {self._path}"
                )
            return self._norm
        else:
            raise ValueError(f"Unknown kind '{kind}'. Use 'flux' or 'norm'.")

    def get_spectra(
        self,
        indices: ArrayLike,
        *,
        kind: str = "flux",
        wave_range: Optional[Tuple[float, float]] = None,
        log10: bool = False,
        squeeze: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch spectra for one or more models.

        Parameters
        ----------
        indices : array-like of int
            Model indices to fetch. Can be a scalar, list, or 1D np.ndarray.
        kind : {"flux", "cont", "norm"}, optional
            "flux" → fluxed spectra from `spectra`.
            "cont" → continuum spectra from `continuua` (if present).
            "norm" → continuum-normalized spectra:
                     - if `continuua` present → spectra / continuua
                     - else if a norm dataset exists → that dataset
        wave_range : (wmin, wmax) or None, optional
            If provided, restrict to that wavelength range (Å).
        log10 : bool, optional
            If True, return log10(spectrum) with a small floor.
        squeeze : bool, optional
            If True, squeeze singleton dimensions (e.g. (1, Nλ) → (Nλ,)).

        Returns
        -------
        wavelengths : np.ndarray, shape (Nλ_sel,)
        spectra : np.ndarray, shape (N_model_sel, Nλ_sel) or (Nλ_sel,) if squeeze
        """
        kind = kind.lower()
        sl_lambda, lam = self._wave_slice(wave_range)

        # Normalize indices to a 1D integer array
        idx = np.atleast_1d(np.asarray(indices, dtype=int))
        if idx.ndim != 1:
            raise ValueError("indices must be 1D or scalar")

        tiny = 1e-30

        if kind == "flux":
            spec = self._flux[idx][:, sl_lambda]

        elif kind == "cont":
            if self._cont is None:
                raise ValueError(
                    "Continuum ('continuua') requested but no 'continuua' dataset "
                    f"is present in {self._path}"
                )
            spec = self._cont[idx][:, sl_lambda]

        elif kind == "norm":
            if self._cont is not None:
                # Compute continuum-normalized spectra on the fly
                f = self._flux[idx][:, sl_lambda]
                c = self._cont[idx][:, sl_lambda]
                spec = f / np.clip(c, tiny, None)
            elif self._norm is not None:
                # Fallback: direct norm dataset, if one exists
                spec = self._norm[idx][:, sl_lambda]
            else:
                raise ValueError(
                    "Normalized spectra requested (kind='norm'), but neither a "
                    "continuum dataset ('continuua') nor a norm dataset is present."
                )
        else:
            raise ValueError("Unknown kind '{kind}'. Use 'flux', 'cont', or 'norm'.")

        spec = np.asarray(spec, dtype=np.float64)

        if log10:
            spec = np.log10(np.clip(spec, tiny, None))

        if squeeze:
            spec = np.squeeze(spec)

        return lam.copy(), spec

    def get_model(
        self,
        *,
        kind: str = "flux",
        wave_range: Optional[Tuple[float, float]] = None,
        log10: bool = False,
        return_index: bool = False,
        **params: float,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, int]]:
        """
        Get the spectrum of the nearest model in label-space to the given params.

        Parameters
        ----------
        kind : {"flux", "norm"}, optional
            Whether to fetch fluxed or normalized spectra.
        wave_range : (wmin, wmax) or None, optional
            Wavelength range in Å.
        log10 : bool, optional
            If True, return log10(spectrum).
        return_index : bool, optional
            If True, also return the index of the matched model.
        params : keyword arguments
            Values for any subset of param_names (e.g. logt, logg, feh, afe, vmic).
            Only these dimensions are used for the distance metric.

        Returns
        -------
        wavelengths, spectrum
        or
        wavelengths, spectrum, index
        """
        if not params:
            raise ValueError("At least one parameter must be provided to get_model")

        # Build label matrix and target vector in the requested dimensions
        fields: List[str] = []
        vals: List[float] = []
        for name, value in params.items():
            if name not in self._param_names:
                raise KeyError(
                    f"Parameter '{name}' not found in grid. "
                    f"Available: {self._param_names}"
                )
            fields.append(name)
            vals.append(float(value))

        # Shape: (N_model, N_dim)
        mat = np.vstack([np.array(self._labels[name], dtype=float) for name in fields]).T
        target = np.array(vals, dtype=float)

        # Euclidean distance in the selected subspace
        diff = mat - target[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        idx_best = int(np.argmin(dist2))

        lam, spec = self.get_spectra(
            idx_best, kind=kind, wave_range=wave_range, log10=log10, squeeze=True
        )

        if return_index:
            return lam, spec, idx_best
        else:
            return lam, spec


# ---------------------------------------------------------------------
# Directory-level interface
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _GridTile:
    """
    Metadata for a single CWC HDF5 grid file living in a larger collection.
    """
    path: str
    feh_min: float
    feh_max: float
    afe_min: float
    afe_max: float
    info: GridInfo   # summary from the underlying CWCFile


class CWCGrid:
    """
    Directory-level interface to a collection of CWC HDF5 grid tiles.

    This class scans a directory for HDF5 files, inspects each one with
    CWCFile to determine its [Fe/H] and [alpha/Fe] coverage, and then
    routes queries to the appropriate file based on the requested
    (feh, afe) point.

    Typical usage
    -------------
    >>> from cwcgrid import CWCGrid
    >>> grid = CWCGrid("/path/to/cwc/h5")

    >>> # Inspect which tiles exist in (FeH, aFe)
    >>> for tile in grid.tiles:
    ...     print(tile.path, tile.feh_min, tile.feh_max, tile.afe_min, tile.afe_max)

    >>> # Get nearest model at given stellar parameters, auto-picking the tile:
    >>> lam, spec = grid.get_model(
    ...     logt=3.76, logg=4.44, feh=0.0, afe=0.0, vmic=1.0,
    ...     kind="flux", wave_range=(3800., 9000.)
    ... )
    """

    def __init__(
        self,
        root_dir: str,
        *,
        pattern: str = "*.h5",
        wave_key: str = "wavelengths",
        labels_key: str = "parameters",
        flux_key: str = "spectra",
        norm_key: Optional[str] = "norm",
        feh_name: str = "feh",
        afe_name: str = "afe",
    ) -> None:
        """
        Parameters
        ----------
        root_dir : str
            Directory containing all the CWC HDF5 files.
        pattern : str, optional
            Glob pattern to select HDF5 files (default: "*.h5").
        wave_key, labels_key, flux_key, norm_key :
            Passed through to each CWCFile instance.
        feh_name : str, optional
            Name of the [Fe/H] field in the labels table.
        afe_name : str, optional
            Name of the [alpha/Fe] field in the labels table.
        """
        root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(root_dir)

        self._root_dir = root_dir
        self._wave_key = wave_key
        self._labels_key = labels_key
        self._flux_key = flux_key
        self._norm_key = norm_key
        self._feh_name = feh_name
        self._afe_name = afe_name

        # Scan all matching HDF5 files and build tiles
        paths = sorted(glob.glob(os.path.join(root_dir, pattern)))
        if not paths:
            raise FileNotFoundError(f"No HDF5 files matching '{pattern}' in {root_dir}")

        tiles: List[_GridTile] = []
        for p in paths:
            # Open each file briefly to read [Fe/H] and [alpha/Fe] ranges
            with CWCFile(
                p,
                wave_key=wave_key,
                labels_key=labels_key,
                flux_key=flux_key,
                norm_key=norm_key,
            ) as g:
                if feh_name not in g.param_names:
                    raise KeyError(
                        f"Field '{feh_name}' not found in labels of {p}. "
                        f"Available: {g.param_names}"
                    )
                if afe_name not in g.param_names:
                    raise KeyError(
                        f"Field '{afe_name}' not found in labels of {p}. "
                        f"Available: {g.param_names}"
                    )

                feh_min, feh_max = g.param_limits(feh_name)
                afe_min, afe_max = g.param_limits(afe_name)

                tiles.append(
                    _GridTile(
                        path=p,
                        feh_min=feh_min,
                        feh_max=feh_max,
                        afe_min=afe_min,
                        afe_max=afe_max,
                        info=g.info,
                    )
                )

        if not tiles:
            raise RuntimeError(f"Found HDF5 files but failed to build any tiles in {root_dir}")

        self._tiles: List[_GridTile] = tiles

        # Lazy cache of the currently open CWCFile instance
        self._current_grid: Optional[CWCFile] = None
        self._current_tile: Optional[_GridTile] = None

    # -----------------------------------------------------------------
    # Basic properties / inspection
    # -----------------------------------------------------------------
    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def tiles(self) -> Tuple[_GridTile, ...]:
        """Tuple of all grid tiles discovered in the directory."""
        return tuple(self._tiles)

    def describe(self) -> None:
        """
        Print a summary of the directory-level collection and its tiles.
        """
        print(f"CWCGrid (directory): {self._root_dir}")
        print(f"  N_tiles = {len(self._tiles)}")
        for i, t in enumerate(self._tiles):
            print(
                f"  Tile {i:02d}: {os.path.basename(t.path)}\n"
                f"    FeH in [{t.feh_min:.3g}, {t.feh_max:.3g}], "
                f"aFe in [{t.afe_min:.3g}, {t.afe_max:.3g}], "
                f"N_models = {t.info.n_models}, "
                f"lambda = [{t.info.lambda_min:.1f}, {t.info.lambda_max:.1f}] Å"
            )

    # -----------------------------------------------------------------
    # Tile selection & grid management
    # -----------------------------------------------------------------
    def _find_tile_index(self, feh: float, afe: float) -> int:
        """
        Return the index of the tile whose (FeH, aFe) box contains the given point.

        Raises ValueError if no tile covers that point.
        """
        for i, t in enumerate(self._tiles):
            if (t.feh_min <= feh <= t.feh_max) and (t.afe_min <= afe <= t.afe_max):
                return i
        raise ValueError(
            f"No grid tile found that covers feh={feh}, afe={afe}. "
            "Check the directory contents or extend the grid."
        )

    def _open_tile(self, tile_index: int) -> CWCFile:
        """
        Ensure that the CWCFile for tile_index is open and cached, and return it.
        """
        tile = self._tiles[tile_index]

        # If the requested tile is already open, just reuse it
        if self._current_tile is not None and self._current_tile.path == tile.path:
            assert self._current_grid is not None
            return self._current_grid

        # Otherwise, close any existing grid and open the new one
        if self._current_grid is not None:
            try:
                self._current_grid.close()
            except Exception:
                pass

        g = CWCFile(
            tile.path,
            wave_key=self._wave_key,
            labels_key=self._labels_key,
            flux_key=self._flux_key,
            norm_key=self._norm_key,
        )
        self._current_grid = g
        self._current_tile = tile
        return g

    def close(self) -> None:
        """Close any open CWCFile."""
        if self._current_grid is not None:
            try:
                self._current_grid.close()
            except Exception:
                pass
        self._current_grid = None
        self._current_tile = None

    def __enter__(self) -> "CWCGrid":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # High-level API: route queries to the correct tile
    # -----------------------------------------------------------------
    def get_model(
        self,
        *,
        kind: str = "flux",
        wave_range: Optional[Tuple[float, float]] = None,
        log10: bool = False,
        return_index: bool = False,
        **params: float,
    ):
        """
        Get the spectrum of the nearest model at a given point in parameter space.

        This is a directory-level analogue of CWCFile.get_model, but it first
        uses the requested feh and afe to choose the correct tile.

        Parameters
        ----------
        kind, wave_range, log10, return_index :
            Same as in CWCFile.get_model.
        params : keyword arguments
            Must include *at least* the fields used for tile selection
            (feh and afe by default), plus any other parameters you want
            to use for the nearest-neighbour search (e.g., logt, logg, vmic).

        Returns
        -------
        (wavelengths, spectrum) or (wavelengths, spectrum, index_within_tile)
        """
        if self._feh_name not in params or self._afe_name not in params:
            raise ValueError(
                f"get_model requires both '{self._feh_name}' and '{self._afe_name}' "
                "in the parameter list so it can choose the correct tile."
            )

        feh_val = float(params[self._feh_name])
        afe_val = float(params[self._afe_name])

        tile_index = self._find_tile_index(feh_val, afe_val)
        grid = self._open_tile(tile_index)

        # Delegate to the per-file CWCFile
        result = grid.get_model(
            kind=kind,
            wave_range=wave_range,
            log10=log10,
            return_index=return_index,
            **params,
        )
        return result

    def get_grid_for(self, feh: float, afe: float) -> CWCFile:
        """
        Convenience method: return the underlying CWCFile for a given (FeH, aFe).

        This lets power users access the full per-tile API (select, get_spectra, etc.).
        """
        tile_index = self._find_tile_index(feh, afe)
        return self._open_tile(tile_index)
    
# ---------------------------------------------------------------------
# End of cwcgrid.py
# ---------------------------------------------------------------------


def demo_cwcfile(filepath: str) -> None:
    print("\n=== Demo: CWCFile (single HDF5 tile) ===")
    with CWCFile(filepath) as g:
        
        g.describe(print_out=True)

        # param limits
        print("logt limits:", g.param_limits("logt"))
        print("feh  limits:", g.param_limits("feh"))

        # nearest model
        lam, spec, idx = g.get_model(
            logt=np.log10(5770.0),
            logg=4.44,
            feh=0.0,
            afe=0.0,
            vmic=1.0,
            kind="flux",
            wave_range=(4000., 7000.),
            return_index=True,
        )
        print("Nearest index within file:", idx)


def demo_cwcgrid(dirpath: str) -> None:
    print("\n=== Demo: CWCGrid (directory router) ===")
    with CWCGrid(dirpath) as grid:
        
        grid.describe()

        # directory-level get_model (auto tile selection)
        lam, spec = grid.get_model(
            logt=np.log10(5770.0),
            logg=4.44,
            feh=0.0,
            afe=0.0,
            vmic=1.0,
            kind="flux",
            wave_range=(4000., 7000.),
        )

        # Drill down into the underlying tile
        g_tile = grid.get_grid_for(feh=0.0, afe=0.0)
        print("Tile file path:", g_tile.info.path)
        idx, pars_sel = g_tile.select(logt=(3.75, 3.78), logg=(4.3, 4.6))
        print("N models in small box:", len(idx))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Describe a CWC HDF5 grid file.")

    parser.add_argument("--filepath", "-f", type=str, default=None, help="Path to an individual CWC HDF5 file.")
    parser.add_argument("--dirpath", "-d", type=str, default=None, help="Path to the directory containing CWC HDF5 files.")

    args = parser.parse_args()

    if args.filepath is not None:
        demo_cwcfile(args.filepath)
    if args.dirpath is not None:
        demo_cwcgrid(args.dirpath)