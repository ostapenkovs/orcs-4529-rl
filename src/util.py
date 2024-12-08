import numpy as np
from numpy import log, exp, sqrt
import matplotlib.pyplot as plt
from scipy.stats import norm

def call(x, k):
    return np.maximum(x-k, 0)

def put(x, k):
    return np.maximum(k-x, 0)

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

def get_mc_price(prices, t1, t2, h, k, r, order=3):
    # Takes as input prices array from call to generate_xxx_paths()
    # t1, t2, r must be identical from aformentioned function call
    # h := payoff function (Callable)
    # k := strike price (int)
    nstep = prices.shape[1]
    dt = (t2 - t1) / nstep  # Time step size

    values = h(prices[:, -1], k)

    for t in range(nstep-2, -1, -1):
        itm = h(prices[:, t], k) > 0
        if not np.any(itm): continue

        X = np.vander(prices[itm, t], order + 1)
        coeff = np.linalg.lstsq(X, exp(-r*dt)*values[itm], rcond=None)[0]

        continuation_values = np.dot(np.vander(prices[:, t], order + 1), coeff)
        exercise_values = h(prices[:, t], k)

        values = np.where(itm, np.maximum(exercise_values, continuation_values), values)

    return exp(-r*dt)*values.mean()

def black_scholes(t1, t2, s, r, q, v, k, call):
    d1 = (log(s/k) + (t2-t1)*(r-q+v*v/2)) / (v*sqrt(t2-t1))
    d2 = d1 - v*sqrt(t2-t1)

    term1 = s*exp(-q*(t2-t1))
    term2 = k*exp(-r*(t2-t1))

    if call:
        return term1*norm.cdf(d1) - term2*norm.cdf(d2)
    return term2*norm.cdf(-d2) - term1*norm.cdf(-d1)

def show_figure(fig, title=None):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    
    if title is not None: fig.suptitle(title, x=0.525, y=1.0725, weight='bold', size='large')

if __name__ == '__main__':
    pass
