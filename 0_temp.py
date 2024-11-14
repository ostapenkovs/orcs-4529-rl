# %%
import numpy as np
from numpy import log, exp, sqrt
import matplotlib.pyplot as plt

seed = 1
rng = np.random.default_rng(seed=seed)

# %%
s_0 = 100; r = 0.05; q = 0.00; v = 0.15; t1 = 0; t2 = 1
nsim = 10000; nstep = 100
dt = (t2 - t1) / nstep

# %%
# dSt = (r-q)St dt + vSt dWt = (r-q)St dt + vSt sqrt(dt) Z
# logSt = logS0 + (r-v**2/2)t + v sqrt(dt) Z
s = np.zeros(shape=(nsim, nstep+1))
s[:, 0] = log(s_0)
s[:, 1:] = (r-q-v*v/2)*dt + v*sqrt(dt)*rng.normal(loc=0, scale=1, size=(nsim, nstep))
s = exp(s.cumsum(axis=1))

# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(s.T)
plt.show()
