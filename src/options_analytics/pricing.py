"""Black-Scholes-Merton European option pricing and Greeks."""

import math

import numpy as np
from scipy.stats import norm


def _validate(S: float, K: float, T: float, sigma: float, option_type: str) -> None:
    """Raise ValueError when any core input is out of its valid domain."""
    if S <= 0:
        raise ValueError(f"S (spot price) must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"K (strike price) must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"T (time to expiry) must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma (implied volatility) must be positive, got {sigma}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    """Return the BSM d1 and d2 intermediate values."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    """Price a European call or put under Black-Scholes-Merton.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset (same currency as K).
    K : float
        Strike price (same currency as S). Must be > 0.
    T : float
        Time to expiry expressed in years (e.g. 0.5 for six months). Must be > 0.
    r : float
        Continuously compounded risk-free interest rate, expressed as a decimal
        (e.g. 0.05 for 5 %).
    sigma : float
        Annualised implied volatility, expressed as a decimal (e.g. 0.20 for 20 %).
        Must be > 0.
    option_type : str
        Either ``"call"`` or ``"put"``.
    q : float, optional
        Continuous dividend yield of the underlying, expressed as a decimal
        (e.g. 0.02 for 2 %). Defaults to 0.0 (no dividends).

    Returns
    -------
    float
        Theoretical option price in the same currency as S and K.

    Raises
    ------
    ValueError
        If S, K, or sigma are non-positive, T is non-positive, or option_type
        is not ``"call"`` or ``"put"``.
    """
    _validate(S, K, T, sigma, option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    discount_S = S * math.exp(-q * T)
    discount_K = K * math.exp(-r * T)

    if option_type == "call":
        return discount_S * norm.cdf(d1) - discount_K * norm.cdf(d2)
    return discount_K * norm.cdf(-d2) - discount_S * norm.cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> dict[str, float]:
    """Compute the five first-order Black-Scholes-Merton Greeks.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset (same currency as K).
    K : float
        Strike price (same currency as S). Must be > 0.
    T : float
        Time to expiry expressed in years. Must be > 0.
    r : float
        Continuously compounded risk-free rate as a decimal.
    sigma : float
        Annualised implied volatility as a decimal. Must be > 0.
    option_type : str
        Either ``"call"`` or ``"put"``.
    q : float, optional
        Continuous dividend yield as a decimal. Defaults to 0.0.

    Returns
    -------
    dict[str, float]
        A dictionary with the following keys and unit conventions:

        ``delta``
            Rate of change of option price with respect to a $1 move in S.
            Dimensionless; call delta ∈ (0, 1), put delta ∈ (-1, 0).

        ``gamma``
            Rate of change of delta with respect to a $1 move in S.
            Units: 1 / currency. Always positive for long options.

        ``vega``
            Rate of change of option price with respect to a **1 percentage
            point** move in volatility (i.e. sigma increases by 0.01).
            Units: currency / vol-point.

        ``theta``
            Rate of change of option price with respect to one **calendar day**
            of time passing (i.e. T decreases by 1/365).
            Units: currency / day. Typically negative for long options.

        ``rho``
            Rate of change of option price with respect to a **1 percentage
            point** move in the risk-free rate (i.e. r increases by 0.01).
            Units: currency / rate-point.

    Raises
    ------
    ValueError
        Same conditions as :func:`bs_price`.
    """
    _validate(S, K, T, sigma, option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    sqrt_T = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    exp_qT = math.exp(-q * T)
    exp_rT = math.exp(-r * T)

    # --- delta ---
    if option_type == "call":
        delta = exp_qT * norm.cdf(d1)
    else:
        delta = exp_qT * (norm.cdf(d1) - 1)

    # --- gamma (identical for call and put) ---
    gamma = exp_qT * pdf_d1 / (S * sigma * sqrt_T)

    # --- vega: scaled to a 1-vol-point (0.01) move ---
    vega = S * exp_qT * pdf_d1 * sqrt_T * 0.01

    # --- theta: scaled to one calendar day (1/365 year) ---
    common_theta = -(S * exp_qT * pdf_d1 * sigma) / (2 * sqrt_T)
    if option_type == "call":
        theta = (
            common_theta
            - r * K * exp_rT * norm.cdf(d2)
            + q * S * exp_qT * norm.cdf(d1)
        ) / 365
    else:
        theta = (
            common_theta
            + r * K * exp_rT * norm.cdf(-d2)
            - q * S * exp_qT * norm.cdf(-d1)
        ) / 365

    # --- rho: scaled to a 1-rate-point (0.01) move ---
    if option_type == "call":
        rho = K * T * exp_rT * norm.cdf(d2) * 0.01
    else:
        rho = -K * T * exp_rT * norm.cdf(-d2) * 0.01

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
