import numpy as np
from numpy import log, exp, sqrt

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

    return exp(-r*dt)*np.mean(values[:, 0])


if __name__ == '__main__':
    pass
