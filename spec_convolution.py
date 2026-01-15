def smoothspec_fft(
    wave_hi: jnp.ndarray,
    flux_hi: jnp.ndarray,
    R_target: float | jnp.ndarray,
    outwave: jnp.ndarray,
    R_in: float | None = None,
    max_half_width: int = 50,   # <- static kernel "radius" in pixels
):
    """
    FFT Gaussian smoothing on a *uniform* high-res grid using resolving power R.

    Parameters
    ----------
    wave_hi : 1D array
        High-res wavelength grid (assumed ~uniform).
    flux_hi : 1D array
        High-res flux on wave_hi.
    R_target : float or 1D array
        Target resolving power R(λ).
    outwave : 1D array
        Output wavelength grid for interpolation.
    R_in : float, optional
        Native resolving power of flux_hi (if not already delta-function).
    max_half_width : int, optional
        Half-width of the convolution kernel in pixels. Kernel length is
        2*max_half_width + 1 and is **static w.r.t. JAX**.

    Returns
    -------
    outflux : 1D array
        Smoothed and resampled flux on outwave.
    """
    wave_hi = jnp.asarray(wave_hi)
    flux_hi = jnp.asarray(flux_hi)
    R_arr   = jnp.asarray(R_target, dtype=wave_hi.dtype)

    # --- pixel scale (no median; just endpoints) ---
    n = wave_hi.shape[0]
    dx = (wave_hi[-1] - wave_hi[0]) / (n - 1)

    # --- convert R to sigma_λ(λ) ---
    sigma_lambda = wave_hi / (R_arr * 2.355)

    if R_in is not None:
        R_in_arr = jnp.asarray(R_in, dtype=wave_hi.dtype)
        sigma_in = wave_hi / (R_in_arr * 2.355)
        sigma_lambda = jnp.sqrt(
            jnp.clip(sigma_lambda**2 - sigma_in**2, 1e-18, jnp.inf)
        )

    # σ in pixels per λ
    sigma_pix = sigma_lambda / dx
    # representative width – can depend on R (traced) since it's only used inside exp
    sigma_pix_eff = jnp.median(sigma_pix)

    # --- STATIC kernel support: length = 2*max_half_width + 1 ---
    half = int(max_half_width)    # make sure this is a Python int, not a jnp array
    k = jnp.arange(-half, half + 1, dtype=wave_hi.dtype)

    # Gaussian kernel with dynamic width
    kernel = jnp.exp(-0.5 * (k / sigma_pix_eff) ** 2)
    kernel = kernel / kernel.sum()

    # --- FFT convolution ---
    n_fft = flux_hi.size + kernel.size - 1
    F = jnp.fft.rfft(flux_hi, n_fft)
    K = jnp.fft.rfft(kernel, n_fft)
    conv = jnp.fft.irfft(F * K, n_fft)

    # center output back onto original grid
    start = (kernel.size - 1) // 2
    conv = conv[start:start + flux_hi.size]

    # resample to outwave
    return jnp.interp(outwave, wave_hi, conv)