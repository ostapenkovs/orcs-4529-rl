import numpy as np
from numpy import log, exp, sqrt
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_gbm_paths(nsim, nstep, t1, t2, s_0, r, q, v, **kwargs):
    # dS_t = mu S_t dt + v S_t dW_t
    rng = kwargs.get('rng', np.random.default_rng())

    dt = (t2 - t1) / nstep

    s = (r-q-v*v/2)*dt + v*sqrt(dt)*rng.normal(loc=0, scale=1, size=(nsim, nstep))
    s = s_0*exp(s.cumsum(axis=1))
    return s

def generate_heston_paths(nsim, nstep, t1, t2, s_0, r, q, v_0, theta, rho, kappa, sigma, **kwargs):
    # dS_t = mu S_t dt + sqrt(v_t) S_t dW_t^1
    # dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^2

    # v_0   := initial variance
    # theta := long-run variance
    # rho   := corr(dW_t^1, dW_t^2)
    # kappa := speed of mean-reversion
    # sigma := vol of vol

    # Feller condition (ensures v_t > 0) := (2*kappa*theta) > sigma**2
    #   can use eps = 1e-8 or similar as failsafe

    # X ~ N(0, 1) and Z = rho*X + sqrt(1-rho**2)*N(0, 1)
    #   corr(X, Z) = rho
    if 2*kappa*theta <= sigma**2: raise ValueError('Feller condition not met in generate_heston_paths.')

    rng = kwargs.get('rng', np.random.default_rng())
    eps = kwargs.get('eps', 1e-8)

    dt = (t2 - t1) / nstep

    s = np.zeros((nsim, nstep))
    for i in range(nsim):
        s_t, v_t = s_0, v_0
        for j in range(nstep):
            z1 = rng.normal(loc=0, scale=1)
            z2 = rho*z1 + sqrt(1-rho**2)*rng.normal(loc=0, scale=1)

            v_t = max(v_t, eps)
            
            s_t += (r-q)*s_t*dt + sqrt(v_t)*s_t*sqrt(dt)*z1
            v_t += kappa*(theta-v_t)*dt + sigma*sqrt(v_t)*sqrt(dt)*z2
            s[i, j] = s_t
    return s

def get_mc_price(prices, t1, t2, h, k, r):
    # Takes as input prices array from call to generate_xxx_paths()
    # t1, t2, r must be identical from aformentioned function call
    # h := payoff function (Callable)
    # k := strike price (int)
    nstep = prices.shape[1]
    dt = (t2 - t1) / nstep

    values = np.zeros_like(prices)
    values[:, -1] = h(prices[:, -1], k)

    for t in range(nstep-2, -1, -1):
        values[:, t] = np.maximum(values[:, t], exp(-r*dt)*values[:, t+1])
        # values[:, t] = np.maximum(values[:, t], exp(-r*dt)*np.mean(values[:, t+1]))

    return exp(-r*dt)*np.mean(values[:, 0])

def black_scholes(t1, t2, s, r, q, v, k, call):
    d1 = (log(s/k) + (t2-t1)*(r-q+v*v/2)) / (v*sqrt(t2-t1))
    d2 = d1 - v*sqrt(t2-t1)

    term1 = s*exp(-q*(t2-t1))
    term2 = k*exp(-r*(t2-t1))

    if call:
        return term1*norm.cdf(d1) - term2*norm.cdf(d2)
    return term2*norm.cdf(-d2) - term1.norm.cdf(-d1)

# def plot_exercise_boundary(agent, env, n_paths=100, strike_price=150):
#     """
#     Visualize the early exercise boundary for a sample of paths.
#     """
#     plt.figure(figsize=(12, 8))
#     exercise_points = []

#     for i in range(n_paths):
#         state = env.reset()
#         path = [state[0]]  # Asset prices
#         times = [0]  # Time steps
#         done = False

#         while not done:
#             # Select an action using a greedy policy
#             action = agent.act(state)

#             if action == 1:  # Exercise
#                 exercise_points.append((env.curr_step * env.dt, state[0]))  # Record exercise point
#                 done = True
#             else:
#                 obs, _, done = env.step(0) 
#                 state = obs  
#                 path.append(state[0]) 
#                 times.append(env.curr_step * env.dt)

#         # Plot the asset price path
#         plt.plot(times, path, color='blue', alpha=0.3)

#     # Plot exercise points if available
#     if exercise_points:
#         ex_times, ex_prices = zip(*exercise_points)
#         plt.scatter(ex_times, ex_prices, color='red', label='Exercise Points')

#     # Add strike price for reference
#     plt.axhline(y=strike_price, color='green', linestyle='--', label='Strike Price')

#     plt.title(f"Early Exercise Boundary Visualization ({n_paths} Paths)")
#     plt.xlabel("Time (Years)")
#     plt.ylabel("Asset Price")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

if __name__ == '__main__':
    pass
