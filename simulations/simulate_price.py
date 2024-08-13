import numpy as np

def generate_price_path(S0, mu, sigma, T, N):
    """
    Generates a price path using Geometric Brownian Motion.

    Args:
        S0: Initial asset price
        mu: Drift
        sigma: Volatility
        T: Time horizon
        N: Number of time steps

    Returns:
        A numpy array of asset prices.
    """

    dt = T / N
    W = np.random.standard_normal(size=N)
    time_steps = np.arange(0, T + dt, dt)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp((mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * W[i - 1])
    return S, time_steps