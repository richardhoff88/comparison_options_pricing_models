import numpy as np
import matplotlib.pyplot as plt

kappa = 4
theta = 0.02
v_0 = 0.02
xi = 0.9
r = 0.02
S = 100
paths = 50000
steps = 2000
T = 1


def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, Npaths, return_vol=False):
    dt = T/steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = np.full(Npaths, S)
    v_t = np.full(Npaths, v_0)
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0, 0]), 
                                           cov=np.array([[1, rho],
                                                         [rho, 1]]), 
                                           size=Npaths) * np.sqrt(dt)
        
        S_t = S_t * np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0])
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t
    
    if return_vol:
        return prices, sigs
    
    return prices


rho = -0.7

# Generate Heston paths
W = generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, paths, return_vol=False)

# Plotting the first 10 simulated paths
plt.plot(W[:10].T)
plt.title('Simulated Stock Price Paths under Heston Model')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()
