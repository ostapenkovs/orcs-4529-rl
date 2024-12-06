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

def get_mc_price(prices, t1, t2, h, r, order = 3):
    # Takes as input prices array from call to generate_xxx_paths()
    # t1, t2, r must be identical from aformentioned function call
    # h := payoff function (Callable)
    # k := strike price (int)
    nsim, nstep = prices.shape
    dt = (t2 - t1) / nstep  # Time step size

    # Option value at maturity (final payoff)
    values = h(prices[:, -1])

    # Work backward through the steps
    for t in range(nstep - 2, -1, -1):
        # Identify paths that are in the money
        itm = h(prices[:, t]) > 0
        if not np.any(itm):
            continue  # No paths in the money, skip this step

        # Get in-the-money prices and discounted future values
        x = prices[itm, t]
        y = np.exp(-r * dt) * values[itm]

        # Fit polynomial regression to continuation values
        X = np.vander(x, order + 1)
        coeff = np.linalg.lstsq(X, y, rcond=None)[0]

        # Estimate continuation values for all paths
        continuation_values = np.dot(np.vander(prices[:, t], order + 1), coeff)

        # Update option values with early exercise logic
        exercise_values = h(prices[:, t])
        values = np.where(itm, np.maximum(exercise_values, continuation_values), values)

    # Discount to present value and return the mean
    return np.exp(-r * t1) * np.mean(values)

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

def black_scholes(S, K, T, r, sigma, option_type="call"):

    if T <= 0:
        # Option has expired
        if option_type == "call":
            return max(S - K, 0)
        elif option_type == "put":
            return max(K - S, 0)
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        # Call option price
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        # Put option price
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")

    return price


if __name__ == '__main__':
    pass
