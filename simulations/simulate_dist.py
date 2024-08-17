import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# from ..counterparty_risk import cva

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def generate_price_path(S0, mu, sigma, T, N):
    """
    Geometric Brownian Motion (GBM)
    Common model for simulating asset prices. 
    It assumes that the asset price follows a log-normal distribution.
    Don't have access to historical financial data.

    Basic equation for GBM:

    dS = S * (mu * dt + sigma * dW)
    
    """
    dt = T / N
    W = np.random.standard_normal(size=N)
    time_steps = np.arange(0, T + dt, dt)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp((mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * W[i - 1])
    return S, time_steps

def simulate_option_prices(S0, K, T, r, sigma, mu, N, num_paths):
    call_prices = []
    for _ in range(num_paths):
        prices, _ = generate_price_path(S0, mu, sigma, T, N)
        call_values = [black_scholes_call(S, K, t, r, sigma) for S, t in zip(prices, np.linspace(0, T, N + 1))]
        call_prices.append(call_values[-1])
    return call_prices


S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
N = 252
mu = 0.08
num_paths = 1000

# simulating
call_prices = simulate_option_prices(S0, K, T, r, sigma, mu, N, num_paths)

print("Mean call price:", np.mean(call_prices))
print("Standard deviation of call prices:", np.std(call_prices))

plt.hist(call_prices, bins=30)
plt.xlabel("Call Price")
plt.ylabel("Frequency")
plt.title("Distribution of Simulated Call Prices")
plt.show()