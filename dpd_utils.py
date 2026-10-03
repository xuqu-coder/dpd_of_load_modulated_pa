"""
Utility functions for a study-oriented digital predistortion notebook

This module intetionally favors clarity over hardware fidelity. It captures
the algorithmic idea from the paper using synthetic PA model and a CRV-insipred
basis expansion that is suitable for local notebook experiments
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray

@dataclass
class Waveform:
    samples: Array
    tx_grid: Array
    active_bins: dict[int, Array]
    carrier_offsets: Array
    n_fft: int
    cp_len: int
    num_symbols: int
    osf: int
    occupied_bins: int
    mask: tuple[int,...]

@dataclass
class CRVConfig:
    linear_memory: int = 2
    nonlinear_orders: tuple[int, ...] = (3,5)
    nonlinear_memory: int = 2
    cross_memory: int = 1
    envelope_lags: tuple[int, ...] = (1, 2)
    include_lead_terms: bool = True
    ridge: float = 1e-6

@dataclass
class DPDModel:
    config: CRVConfig
    coeffs: Array
    basis_names: list[str]
    active_columns: Array

def db10(x: Array | float) -> Array | float:
    return 10.0 * np.log10(np.maximum(np.asarray(x),1e-15))

def rms(x: Array) -> float:
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))

def normalize_power(x: Array, target_power: float=1.0) -> Array:
    power = np.mean(np.abs(x) ** 2)
    if power <= 0:
        return x.copy()
    return x * np.sqrt(target_power / power)

def qam16_modulate(symbol_ids: Array) -> Array:
    levels = np.array([-3, -1, 1, 3], dtype = float)
    i_gray = np.array([0, 1, 3, 2])
    q_gray = np.array([0, 1, 3, 2])
    i_idx = i_gray[(symbol_ids >> 2) & 0x3]
    q_idx = q_gray[symbol_ids & 0x2]
    constellation = levels[i_idx] + 1j * levels[q_idx]
    return constellation / np.sqrt(np.mean(np.abs(constellation ** 2)))

def make_rng(seed: int=0) -> np.random.Generator:
    return np.random.default_rng(seed)

def generate_multicarrier_ofdm(
    active_mask: Iterable[int],
    rng: np.random.Generator,
    n_fft: int = 256,
    cp_len: int = 32,
    num_symbols: int = 24,
    osf: int = 4,
    occupied_bins: int = 36,
    spacing_bins: int = 96,
) -> Waveform:
    mask = tuple(int(v) for v in active_mask)
    if len(mask) != 5:
        raise ValueError("active_mask must contain five carrier flags")
    # Multicarrier initialization
    n_fft_hi = n_fft * osf
    cp_hi = cp_len * osf
    center = n_fft_hi // 2
    shfited_grid = np.zeros((num_symbols, n_fft_hi), dtype=complex)
    carrier_offsets = np.array([-2, -1, 0, 1, 2]) * spacing_bins
    half_occ = occupied_bins // 2
    active_bins: dict[int, Array] = {}

    # Generate each carriers' samples and stored in shifted_grid
    for carrier_idx, enabled in enumerate(mask):
        if not enabled:
            continue
        # current carrier's offset
        offset = carrier_offsets[carrier_idx]
        # occupied bins of current carriers
        bins = center + offset + np.arange(-half_occ, half_occ)
        # generate symbols ID for current carreir
        symbol_ids = rng.integers(0, 16, size=(num_symbols, occupied_bins))
        # map symbols ID to QAM constellation
        qam = qam16_modulate(symbol_ids)
        # store samples in the grid
        # [row, col]: row is OFDM symbols, col is FFT bins/subcarriers
        shfited_grid[:, bins] = qam
        active_bins[carrier_idx] = bins

    time_no_cp = np.fft.ifft(np.fft.fftshift(shfited_grid, axes=1),axis=1)
    time_no_cp *= np.sqrt(n_fft_hi)
    with_cp = np.concatenate([time_no_cp[:,-cp_hi:], time_no_cp],axis=1)
    samples = normalize_power(with_cp.reshape(-1))

    return Waveform(
        samples=samples,
        tx_grid=shfited_grid,
        active_bins=active_bins,
        carrier_offsets=carrier_offsets,
        n_fft=n_fft_hi,
        cp_len=cp_hi,
        num_symbols=num_symbols,
        osf=osf,
        occupied_bins=occupied_bins,
        mask=mask,
    )

def reshape_symbols(x: Array, n_fft: int, cp_len: int) -> Array:
    symbol_len = n_fft + cp_len
    usable = ((len(x)) // symbol_len) * symbol_len
    trimmed = x[:usable]
    return trimmed.reshape(-1, symbol_len)

def waveform_to_grid(x:Array, meta: Waveform) -> Array:
    symbols = reshape_symbols(x, meta.n_fft, meta.cp_len)
    no_cp = symbols[:, meta.cp_len :]
    shifted_grid = np.fft.fftshift(np.fft.fft(no_cp, axis=1), axes=1) / np.sqrt(meta.n_fft)
    return shifted_grid

def measure_papr_db(x: Array) -> float:
    peak = np.max(np.abs(x) ** 2)
    avg = np.mean(np.abs(x) ** 2)
    return float(db10(peak / max(avg, 1e-15)))

def best_linear_gain(y: Array, x_ref: Array) -> complex:
    denom = np.vdot(y, y).real
    if denom <= 1e-15:
        return 1.0 + 0.0j
    return np.vdot(y, x_ref)/denom

def nmse_db(y: Array, x_ref: Array) -> float:
    gain = best_linear_gain(y, x_ref)
    err = gain * y - x_ref
    return float(db10(np.mean(np.abs(err) **2) / max(np.mean(np.abs(x_ref) ** 2), 1e-15) ))

def evm_percent(y: Array, meta: Waveform) -> float:
    rx_grid = waveform_to_grid(y, meta)
    ref_grid = meta.tx_grid[: rx_grid.shape[0], :]
    err_power = 0.0
    ref_power = 0.0
    for bins in meta.active_bins.values():
        ref = ref_grid[:, bins]
        rx = rx_grid[:, bins]
        denom = np.sum(np.abs(rx) ** 2)
        if denom <= 1e-15:
            equalizer = 1.0 + 0.0j
        else:
            equalizer = np.sum(ref * np.conj(rx))/denom
        err = equalizer * rx - ref
        err_power += np.sum(np.abs(err) ** 2)
        ref_power += np.sum(np.abs(ref) ** 2)

    if ref_power <= 1e-15:
        return 0.0
    
    return float(100.0 * np.sqrt(err_power / ref_power))

def fft_power_spectrum(x: Array, n_fft: int = 16384) -> tuple[Array, Array]:
    n = max(n_fft, int(2 ** np.ceil(np.log2(len(x) + 1))))
    spec = np.fft.fftshift(np.fft.fft(x, n=n))
    power = np.abs(spec) ** 2 / n
    freq = np.linspace(-0.5, 0.5, n, endpoint=False)
    return freq, power

def worst_acpr_db(x: Array, meta: Waveform, guard_bins: int = 8) -> float:
    freq, power = fft_power_spectrum(x)
    bin_width = 1.0 / meta.n_fft
    occ_half = meta.occupied_bins / (2.0 * meta.n_fft)
    adj_half = meta.occupied_bins / (2.0 * meta.n_fft)
    worst = -300.0

    for carrier_idx, bins in meta.active_bins.items():
        center = meta.carrier_offsets[carrier_idx] / meta.n_fft
        inband = np.abs(freq - center) <= occ_half
        left_adj = np.abs(freq - (center - (2*occ_half + guard_bins * bin_width))) <= adj_half
        right_adj = np.abs(freq - (center + (2 * occ_half + guard_bins * bin_width))) <= adj_half

        p_in = np.sum(power[inband])
        p_left = np.sum(power[left_adj])
        p_right = np.sum(power[right_adj])

        if p_in <= 1e-15:
            continue

        worst = max(worst, float(db10(max(p_left, p_right) / p_in)))
    return worst

def shift_signal(x:Array, lag: int) -> Array:
    out = np.zeros_like(x)
    if lag == 0:
        out[:] = x
    elif lag >0 :
        out[lag:] = x[:-lag]
    else:
        out[:lag] = x[-lag:]
    return out

def build_crv_basis(x: Array, config: CRVConfig) -> tuple[Array, list[str]]:
    columns: list[Array] = []
    names: list[str] = []

    for mem in range(config.linear_memory + 1):
        xm = shift_signal(x, mem)
        columns.append(xm)
        names.append(f"linear[m={mem}]")

    for order in config.nonlinear_orders:
        for mem in range(config.nonlinear_memory + 1):
            xm = shift_signal(x, mem)
            columns.append(xm * np.abs(xm) ** (order - 1))
            names.append(f"aligned[p={order},m={mem}]")

    for order in config.nonlinear_orders:
        if order < 3:
            continue
        env_power = order - 1
        for mem in range(config.cross_memory + 1):
            base = shift_signal(x, mem)
            for lag in config.envelope_lags:
                lag_env = shift_signal(x, mem+lag)
                columns.append(base * np.abs(lag_env) ** env_power)
                names.append(f"lag[p={order},m={mem},k={lag}]")
                if config.include_lead_terms:
                    lead_env = shift_signal(x, mem - lag)
                    columns.append(base * np.abs(lead_env) ** env_power)
                    names.append(f"lead[p={order},m={mem},k={lag}]")

    basis = np.column_stack(columns)
    return basis, names

def solve_ridge(Phi: Array, target:Array, ridge: float) -> Array:
    # solve the ridge regression problem: min((|w*y - xref|^2 + ridge * |w|^2)) for w(coefficients)
    #   w = (y^H * y + ridge * I)^-1 * y^H * xref
    #   (y^H * y + ridge * I) w = y^H * xref
    #   (gram + reg) w = rhs
    # ridge is helpful when
    #   1. matrix is not full rank
    #   2. number of features (basis) is large
    #   3. features are correlated
    #   4. data is noisy
    #   5. more robust estimation as opposed to exact fit on the training samples
    gram = Phi.conj().T @ Phi
    rhs = Phi.conj().T @ target
    reg = ridge * np.eye(gram.shape[0], dtype=gram.dtype)
    return np.linalg.solve(gram + reg, rhs)

def lowpass_frequency_domain(x:Array, normalized_bw: float) -> Array:
    """This function keeps only frequency content with normalized frequency magnitude <= normalized_bw
    and removes the rest by zeroing those FFT bins"""
    if not 0.0 < normalized_bw <=0.5:
        raise ValueError("normalized_bw must lie in (0, 0.5]")
    n = len(x)
    spec = np.fft.fftshift(np.fft.fft(x))
    freq = np.linspace(-0.5, 0.5, n, endpoint = False)
    mask = np.abs(freq) <= normalized_bw
    filtered = np.fft.ifft(np.fft.ifftshift(spec * mask))
    return filtered

def filter_basis_columns(Phi: Array, normalized_bw: float) -> Array:
    filtered = np.empty_like(Phi)
    for idx in range(Phi.shape[1]):
        filtered[:, idx] = lowpass_frequency_domain(Phi[:, idx], normalized_bw)
    return filtered

def fit_indirect_model(
    x_target: Array,
    y_observed: Array,
    config: CRVConfig,
    observation_bw: float | None = None,
    basis_bandlimited_aware: bool = True,
    prune_threshold_db: float | None = None,
) -> tuple[DPDModel, dict[str, Array | float]]:
    
    # source is the observed signal
    source = y_observed
    # naive path: filter the source signal to the observation window before building the
    # basis (mismatched, since nonlinear terms of a filtered signal are not the filtered
    # nonlinear terms). The bandwidth-aware path instead filters the basis columns below.
    if observation_bw is not None and not basis_bandlimited_aware:
        source = lowpass_frequency_domain(source, observation_bw)
    # build the CRV basis function
    Phi, names = build_crv_basis(source, config)
    # filter the basis functions to the observation bandwidth
    if observation_bw is not None and basis_bandlimited_aware:
        Phi = filter_basis_columns(Phi, observation_bw)
    # solve for the coefficients w by solving the ridge regression problem
    coeffs = solve_ridge(Phi, x_target, config.ridge)
    # get the active columns (basis function)
    active = np.arange(Phi.shape[1])
    # calculate the contribution of each basis function in dB, power of each basis function over the peak power
    # this is used to prune weak basis function to simplifiy the model and remove negligible terms
    contrib_db = contribution_db(Phi, coeffs)
    if prune_threshold_db is not None:
        active = np.where(contrib_db >= prune_threshold_db)[0]
        if active.size == 0:
            active = np.array([0])
        coeffs = solve_ridge(Phi[:,active], x_target, config.ridge)
        names = [names[i] for i in active]
        contrib_db = contribution_db(Phi[:, active], coeffs)
    else:
        names = list(names)

    model = DPDModel(config=config, coeffs=coeffs,basis_names=names, active_columns=active)
    info = {
        "contribution_db": contrib_db,
        "num_coeffs": float(len(coeffs)),
    }
    return model, info

def apply_dpd(x: Array, model:DPDModel) -> Array:
    Phi, _ = build_crv_basis(x, model.config)
    Phi = Phi[:, model.active_columns]
    return Phi @ model.coeffs

def contribution_db(Phi: Array, coeffs: Array) -> Array:
    """This function calculates how strong each model term is, relative to the strongest term and express it in dB"""
    power = np.mean(np.abs(Phi * coeffs[None, :]) ** 2, axis=0)
    peak = np.max(power)
    if peak <= 1e-15:
        return -300.0 * np.ones_like(power)
    return db10(power / peak)

def make_synthetic_pa() -> Callable[[Array], Array]:
    def pa(x:Array) -> Array:
        x0 = x
        x1 = shift_signal(x0, 1)
        x2 = shift_signal(x0, 2)
        y = (
            0.9 * x0
            + 0.12 * x1
            + (0.2 - 0.08j) * x0 * np.abs(x0) ** 2
            + (0.03 + 0.04j) * x1 * np.abs(x1) ** 2
            + (0.02 - 0.03j) * x2 * np.abs(x0) ** 2
            + 0.015 * x0 * np.abs(x2) ** 2
        )
        y *= np.exp(1j * 0.08 * np.abs(x0)**2)
        return y
    return pa

def make_linear_pa(gain: complex = 1.0 + 0.0j) -> Callable[[Array], Array]:
    def pa(x: Array) -> Array:
        return gain * x
    return pa

def direct_learning_update(
    x_ref: Array,
    pa_fn: Callable[[Array], Array],
    model: DPDModel,
    step_size: float = 0.65,
) -> DPDModel:
    Phi, _ = build_crv_basis(x_ref, model.config)
    Phi = Phi[:, model.active_columns]
    z = Phi @ model.coeffs
    y = pa_fn(z)
    gain = best_linear_gain(y, x_ref)
    error = x_ref - gain * y
    delta = solve_ridge(Phi, error, model.config.ridge)
    updated = DPDModel(
        config=model.config,
        coeffs=model.coeffs + step_size * delta,
        basis_names=model.basis_names,
        active_columns=model.active_columns,
    )
    return updated

def evaluate_chain(
    x_ref: Array,
    meta: Waveform,
    pa_fn: Callable[[Array], Array],
    model: DPDModel | None = None,
) -> dict[str, float]:
    # Apply DPD to the input signal
    z = x_ref if model is None else apply_dpd(x_ref, model)
    # Apply PA model to the DPD output
    y = pa_fn(z)
    # calculate the best linear gain
    #   solveing g * y ~= x_ref by least squares (min(sum( |g*y - xref|^2))) for g
    #   g = (y^H*y)^-1 * y^H * x_ref = <g, x_ref> / <y, y>
    gain = best_linear_gain(y, x_ref)
    aligned = gain * y
    metrics = {
        "papr_db": measure_papr_db(z),
        "acpr_db": worst_acpr_db(aligned, meta),
        "evm_pct": evm_percent(aligned, meta),
        "nmse_db": nmse_db(y, x_ref),
        "num_coeffs": 0.0 if model is None else float(len(model.coeffs)),
    }
    for value in metrics.values():
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite metric encountered")
    return metrics

def run_learning_workflow(
    waveform: Waveform,
    pa_fn: Callable[[Array], Array],
    config: CRVConfig,
    indirect_iters: int = 2,
    direct_iters: int = 2,
    observation_bw: float | None = None,
    basis_bandlimit_aware: bool = True,
    prune_threshold_db: float | None = None,
    direct_step_size: float = 0.65,
) -> typle[DPDModel, list[dict[str, float]]]:
    x = waveform.samples
    model: DPDModel | None = None
    history: list[dict[str, float]] = []

    history.append({"iteration": 0.0, "stage":"baseline", **evaluate_chain(x, waveform, pa_fn, None)})

    for idx in range(indirect_iters):
        z = x if model is None else apply_dpd(x, model)
        y = pa_fn(z)
        model, _ = fit_indirect_model(
            x_target=x,
            y_observed=y,
            config=config,
            observation_bw=observation_bw,
            basis_bandlimited_aware=basis_bandlimit_aware,
            prune_threshold_db=prune_threshold_db,
        )
        history.append(
            {
                "iteration": float(idx + 1),
                "stage": "indirect",
                **evaluate_chain(x, waveform, pa_fn, model),
            }
        )

    if model is None:
        model, _ = fit_indirect_model(x, pa_fn(x), config)

    for idx in range(direct_iters):
        model = direct_learning_update(x, pa_fn, model, step_size=direct_step_size)
        history.append(
            {
                "iteration": float(indirect_iters + idx + 1),
                "stage": "direct",
                **evaluate_chain(x, waveform, pa_fn, model),
            }
        )

    return model, history

def concatenate_waveforms(waveforms: list[Waveform]) -> Waveform:
    if not waveforms:
        raise ValueError("waveforms cannot be empty")
    first = waveforms[0]
    concat_samples = np.concatenate([wf.samples for wf in waveforms])
    concat_grid = np.concatenate([wf.tx_grid for wf in waveforms], axis = 0)
    return Waveform(
        samples=concat_samples,
        tx_grid=concat_grid,
        active_bins=first.active_bins,
        carrier_offsets=first.carrier_offsets,
        n_fft=first.n_fft,
        cp_len=first.cp_len,
        num_symbols=sum(wf.num_symbols for wf in waveforms),
        osf = first.osf,
        occupied_bins=first.occupied_bins,
        mask = first.mask,
    )

def train_general_model(
    train_waveforms: list[Waveform],
    pa_fn: Callable[[Array], Array],
    config: CRVConfig,
    indirect_iters: int = 3,
    direct_iters: int = 3,
    prune_threshold_db: float | None = None,
    observation_bw: float | None = None,
    basis_bandlimit_aware: bool = True,
) -> tuple[DPDModel, list[dict[str, float]]]:
    x_concat = np.concatenate([wf.samples for wf in train_waveforms])
    meta_for_history = train_waveforms[0]
    model: DPDModel | None = None
    history: list[dict[str, float]] = []

    baseline_metrics = evaluate_chain(train_waveforms[0].samples, train_waveforms[0], pa_fn, None)
    history.append({"iteration":0.0, "stage": "baseline", **baseline_metrics})

    for idx in range(indirect_iters):
        z = x_concat if model is None else apply_dpd(x_concat, model)
        y = pa_fn(z)
        model, _ = fit_indirect_model(
            x_target = x_concat,
            y_observed=y,
            config=config,
            observation_bw=observation_bw,
            basis_bandlimited_aware=basis_bandlimit_aware,
            prune_threshold_db=prune_threshold_db
        )
        history.append(
            {
                "iteration": float(idx + 1),
                "stage": "indirect",
                **evaluate_chain(meta_for_history.samples, meta_for_history, pa_fn, model),
            }
        )
    
    if model is None:
        model, _ = fit_indirect_model(x_concat, pa_fn(x_concat), config)

    for idx in range(direct_iters):
        model = direct_learning_update(x_concat, pa_fn, model, step_size=0.60)
        history.append(
            {
                "iteration": float(indirect_iters + idx + 1),
                "stage": "direct",
                **evaluate_chain(meta_for_history.samples, meta_for_history, pa_fn, model),
            }
        )

    return model, history

def clipping_filtering(x: Array, meta: Waveform, clip_db: float, iterations: int = 4) -> Array:
    """Iterative clipping and filtering for crest-factor reduction.

    The envelope is clipped at clip_db above the mean power, then the clipping noise
    is removed outside the active subcarriers. Filtering is done per OFDM symbol on
    the transmitter's FFT grid, so an unclipped signal passes through unchanged.
    Filtering regrows the peaks, so the clip/filter pair is repeated a few times.
    """
    keep = np.zeros(meta.n_fft, dtype=bool)
    for bins in meta.active_bins.values():
        keep[bins] = True
    keep = np.fft.ifftshift(keep)
    threshold = np.sqrt(10 ** (clip_db / 10.0) * np.mean(np.abs(x) ** 2))
    out = x.copy()
    for _ in range(iterations):
        mag = np.abs(out)
        over = mag > threshold
        out[over] *= threshold / np.maximum(mag[over], 1e-15)
        body = reshape_symbols(out, meta.n_fft, meta.cp_len)[:, meta.cp_len :]
        spec = np.fft.fft(body, axis=1)
        spec[:, ~keep] = 0.0
        body = np.fft.ifft(spec, axis=1)
        out = np.concatenate([body[:, -meta.cp_len :], body], axis=1).reshape(-1)
    return normalize_power(out)

def make_saturating_pa(
    sat: float = 1.0,
    smoothness: float = 2.0,
    am_pm_rad: float = 0.5,
    memory: tuple[complex, ...] = (1.0, 0.12, -0.04),
) -> Callable[[Array], Array]:
    """Wiener PA: short FIR memory followed by Rapp AM/AM and a saturating AM/PM.

    Unlike make_synthetic_pa, the output amplitude is bounded by sat, so peaks that
    drive past saturation cannot be recovered by DPD. This is what makes PAPR matter.
    """
    taps = np.asarray(memory, dtype=complex)
    def pa(x: Array) -> Array:
        u = sum(tap * shift_signal(x, lag) for lag, tap in enumerate(taps))
        r = np.abs(u) / sat
        am = 1.0 / (1.0 + r ** (2 * smoothness)) ** (1.0 / (2 * smoothness))
        pm = am_pm_rad * r ** 2 / (1.0 + r ** 2)
        return am * u * np.exp(1j * pm)
    return pa

def regrowth_acpr_db(y: Array, x_ref: Array, meta: Waveform, guard_bins: int = 8) -> float:
    """Worst adjacent-channel power of the PA-generated distortion only.

    worst_acpr_db measures the whole output, which for this unwindowed OFDM is floored
    near -25 dB by the input's own sidelobes. Here the reference is subtracted first, so
    the adjacent-channel power counts only what the PA (and DPD residual) added.
    """
    err = best_linear_gain(y, x_ref) * y - x_ref
    freq, p_err = fft_power_spectrum(err)
    _, p_ref = fft_power_spectrum(x_ref)
    half = meta.occupied_bins / (2.0 * meta.n_fft)
    step = 2 * half + guard_bins / meta.n_fft
    worst = -300.0
    for carrier_idx in meta.active_bins:
        center = meta.carrier_offsets[carrier_idx] / meta.n_fft
        p_in = np.sum(p_ref[np.abs(freq - center) <= half])
        p_adj = max(np.sum(p_err[np.abs(freq - (center + s * step)) <= half]) for s in (-1, 1))
        worst = max(worst, float(db10(p_adj / max(p_in, 1e-15))))
    return worst

def summarize_models(
    label_to_metrics: dict[str, dict[str, float]]
) -> tuple[list[str], Array]:
    labels = list(label_to_metrics.keys())
    data = np.array(
        [
            [
                label_to_metrics[label]["acpr_db"],
                label_to_metrics[label]["evm_pct"],
                label_to_metrics[label]["nmse_db"],
                label_to_metrics[label]["num_coeffs"],
            ]
            for label in labels
        ]
    )
    return labels, data

def plot_spectrum(x: Array, label: str, ax: plt.Axes | None = None) -> plt.Axes:
    freq, power = fft_power_spectrum(x)
    psd_db = db10(power / np.max(power))
    if ax is None:
        _, ax = plt.subplots(figsize=(9,4))
    ax.plot(freq, psd_db, label=label)
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Relative PSD (dB)")
    ax.grid(True, alpha=0.3)
    return ax

def plot_learning_history(history: list[dic[str, float]]) -> plt.Figure:
    iterations = [item["iteration"] for item in history]
    acpr = [item["acpr_db"] for item in history]
    evm = [item["evm_pct"] for item in history]
    nmse = [item["nmse_db"] for item in history]
    coeffs= [item["num_coeffs"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(11,7), constrained_layout = True)
    axes[0, 0].plot(iterations, acpr, marker="o")
    axes[0, 0].set_title("Worst ACPR")
    axes[0, 0].set_ylabel("dB")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iterations, evm, marker="o")
    axes[0, 1].set_title("EVM")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iterations, nmse, marker="o")
    axes[1, 0].set_title("NMSE")
    axes[1, 0].set_ylabel("dB")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iterations, coeffs, marker="o")
    axes[1, 1].set_title("Active coefficients")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].grid(True, alpha=0.3)

    return fig


def compare_models_on_waveforms(
    models: dict[str, DPDModel | None],
    waveforms: list[Waveform],
    pa_fn: Callable[[Array], Array],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for wf in waveforms:
        label = "".join(str(v) for v in wf.mask)
        for model_name, model in models.items():
            metrics = evaluate_chain(wf.samples, wf, pa_fn, model)
            rows.append(
                {
                    "waveform": label,
                    "model": model_name,
                    **metrics,
                }
            )
    return rows


def rows_to_table(rows: list[dict[str, float | str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append(" | ".join(values))
    return "\n".join(lines)
