import sys, os
import pickle
import time
import numpy as np
from numpy import log, exp, sqrt
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
import multiprocessing as mp

sys.path.append(os.path.join(os.getcwd(), '..'))
from src.model import Environment, Agent

COLS = ('s_0', 'v', 't2', 'euro', 'amer', 'se', 'early_exercise_value')

def read_data():
    l = [[] for _ in range(len(COLS))]

    with open('../resources/price_table.txt', 'r') as f:
        for i, line in enumerate(f):
            val = float(line.strip())
            l[ i % len(COLS) ].append(val)
    
    l = [l[i] for i in [0, 1, 2, 4]]
    return l

def call(x, k):
    return np.maximum(x-k, 0)

def put(x, k):
    return np.maximum(k-x, 0)

def onesim(*args, h, nsim, nstep, t1, r, q, k):
    s_0, v, t2, eval_price = args
    path_kwargs = dict(v=v)

    env = Environment(
        nsim=nsim, nstep=nstep, t1=t1, t2=t2, s_0=s_0, r=r, q=q,
        path_kwargs=path_kwargs, h=h, k=k, gbm=True
    )

    agent = Agent(
        env=env, hidden_dim=128, depth=3, lr=0.001, buffer_size=1024, batch_size=64,
        buffer_interval=8, model_interval=32, gamma=0.99, eps=0.99, eps_decay=0.995, eps_min=0.01
    )

    losses, rewards, fig1 = agent.train(nepisode=3, notebook=False, verbose=False)
    mean_reward, fig2 = agent.eval(nepisode=3, notebook=False)
    mse = (mean_reward - eval_price)**2

    return losses, rewards, mean_reward, eval_price, mse, fig1, fig2

def main():
    ### DATA ###
    l = read_data()
    print('Need to run:', len(l[0]), 'simulations.')
    print()
    ### DATA ###

    ### MULTIPROCESS ###
    num_workers = mp.cpu_count()
    print('Num. workers:', num_workers)
    print()

    start = time.time()
    
    pool = mp.Pool(processes=num_workers)
    myfunc = partial(onesim, h=put, nsim=10000, nstep=365, t1=0, r=0.04, q=0.00, k=100)
    results = pool.starmap(myfunc, zip(*l))
    pool.close()
    pool.join()
    
    elapsed = time.time() - start
    ### MULTIPROCESS ###

    ### RESULTS ###
    print(f'Elapsed: { round(elapsed / 60, 4) } min.')
    
    with open('../data/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    ### RESULTS ###

if __name__ == '__main__':
    if sys.platform == 'darwin': mp.set_start_method('spawn')
    main()
