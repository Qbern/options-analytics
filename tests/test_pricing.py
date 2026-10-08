"""Tests for the Black-Scholes-Merton pricing engine."""

import math

import pytest

from options_analytics.pricing import bs_greeks, bs_price

# Reference parameters (ATM, 1-year, no dividends)
REF = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call")


class TestBsPrice:
    def test_call_reference_value(self):
        price = bs_price(**REF)
        assert abs(price - 10.4506) < 1e-4

    def test_put_reference_value(self):
        params = {**REF, "option_type": "put"}
        price = bs_price(**params)
        assert abs(price - 5.5735) < 1e-4

    def test_put_call_parity_no_dividends(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = bs_price(S, K, T, r, sigma, "call", q=0.0)
        put = bs_price(S, K, T, r, sigma, "put", q=0.0)
        lhs = call - put
        rhs = S * math.exp(0.0) - K * math.exp(-r * T)  # q=0
        assert abs(lhs - rhs) < 1e-10

    def test_put_call_parity_with_dividends(self):
        S, K, T, r, sigma, q = 100.0, 95.0, 0.5, 0.03, 0.25, 0.02
        call = bs_price(S, K, T, r, sigma, "call", q=q)
        put = bs_price(S, K, T, r, sigma, "put", q=q)
        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10


class TestBsGreeks:
    def test_call_delta_in_range(self):
        greeks = bs_greeks(**REF)
        assert 0.0 < greeks["delta"] < 1.0

    def test_put_delta_in_range(self):
        greeks = bs_greeks(**{**REF, "option_type": "put"})
        assert -1.0 < greeks["delta"] < 0.0

    def test_gamma_positive(self):
        greeks = bs_greeks(**REF)
        assert greeks["gamma"] > 0.0

    def test_vega_positive(self):
        greeks = bs_greeks(**REF)
        assert greeks["vega"] > 0.0

    def test_call_and_put_share_same_gamma(self):
        call_g = bs_greeks(**REF)["gamma"]
        put_g = bs_greeks(**{**REF, "option_type": "put"})["gamma"]
        assert abs(call_g - put_g) < 1e-12

    def test_call_and_put_share_same_vega(self):
        call_v = bs_greeks(**REF)["vega"]
        put_v = bs_greeks(**{**REF, "option_type": "put"})["vega"]
        assert abs(call_v - put_v) < 1e-12

    def test_vega_unit_one_vol_point(self):
        """vega ≈ price(sigma+0.01) - price(sigma)."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        vega = bs_greeks(S, K, T, r, sigma, "call")["vega"]
        bump = bs_price(S, K, T, r, sigma + 0.01, "call") - bs_price(S, K, T, r, sigma, "call")
        assert abs(vega - bump) < 0.001

    def test_theta_unit_one_calendar_day(self):
        """theta ≈ price(T-1/365) - price(T)."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        theta = bs_greeks(S, K, T, r, sigma, "call")["theta"]
        bump = bs_price(S, K, T - 1 / 365, r, sigma, "call") - bs_price(S, K, T, r, sigma, "call")
        assert abs(theta - bump) < 0.001

    def test_rho_unit_one_rate_point(self):
        """rho ≈ central finite-difference scaled to a 1-rate-point (0.01) move."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        h = 1e-4  # small step avoids convexity error from the full 0.01 bump
        rho = bs_greeks(S, K, T, r, sigma, "call")["rho"]
        bump = (
            bs_price(S, K, T, r + h, sigma, "call")
            - bs_price(S, K, T, r - h, sigma, "call")
        ) / (2 * h) * 0.01
        assert abs(rho - bump) < 0.001


class TestInputValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(S=-1.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call"),
            dict(S=100.0, K=0.0, T=1.0, r=0.05, sigma=0.20, option_type="call"),
            dict(S=100.0, K=100.0, T=-0.1, r=0.05, sigma=0.20, option_type="call"),
            dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.0, option_type="call"),
            dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="straddle"),
        ],
    )
    def test_bs_price_raises_value_error(self, kwargs):
        with pytest.raises(ValueError):
            bs_price(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(S=0.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="put"),
            dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=-0.1, option_type="put"),
            dict(S=100.0, K=100.0, T=0.0, r=0.05, sigma=0.20, option_type="put"),
        ],
    )
    def test_bs_greeks_raises_value_error(self, kwargs):
        with pytest.raises(ValueError):
            bs_greeks(**kwargs)
