import sys, os
import pickle
import time
import numpy as np
from numpy import log, exp, sqrt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

from functools import partial
import multiprocessing as mp

sys.path.append(os.path.join(os.getcwd(), '..'))
from src.model import Environment, Agent

def call(x, k):
    return np.maximum(x-k, 0)

def put(x, k):
    return np.maximum(k-x, 0)

def read_data(data_dir='../data/test_cases.csv'):
    cols = ['s_0', 't2', 'q', 'r', 'h']
    gbmcols = ['v']
    hestoncols = ['v_0', 'theta', 'rho', 'kappa', 'sigma']

    df = pd.read_csv(data_dir, usecols=range(0, 14))

    df['h'] = df['h'].apply(lambda x: call if x == 'American_Call' else put)

    df['path_kwargs'] = np.where(df['gbm'], df[gbmcols].to_dict('records'), df[hestoncols].to_dict('records'))

    df.drop(gbmcols + hestoncols, axis=1, inplace=True)

    return df[cols + ['path_kwargs', 'gbm']].values

def onesim(*args, nsim, nstep, t1, k):
    s_0, t2, q, r, h, path_kwargs, gbm = args

    env = Environment(
        nsim=nsim, nstep=nstep, t1=t1, t2=t2, s_0=s_0, r=r, q=q,
        path_kwargs=path_kwargs, h=h, k=k, gbm=gbm
    )

    agent = Agent(
        env=env, hidden_dim=128, depth=3, lr=0.001, buffer_size=1024, batch_size=64,
        buffer_interval=8, model_interval=32, gamma=0.995, eps=0.99, eps_decay=0.995, eps_min=0.01
    )

    losses, rewards, fig1 = agent.train(nepisode=10000, notebook=False, verbose=False)
    mean_reward, fig2 = agent.eval(nepisode=1000, notebook=False)

    return mean_reward, fig1, fig2

def main():
    ### DATA ###
    arr = read_data(data_dir='../data/test_cases.csv')
    print('Need to run:', arr.shape[0], 'simulations.')
    print()
    ### DATA ###

    ### MULTIPROCESS ###
    num_workers = mp.cpu_count()
    print('Num. workers:', num_workers)
    print()

    start = time.time()
    
    pool = mp.Pool(processes=num_workers)
    myfunc = partial(onesim, nsim=10000, nstep=52, t1=0, k=100)
    results = pool.starmap(myfunc, arr)
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
