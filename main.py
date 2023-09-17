import os
from collections import Counter
from joblib import Parallel, delayed
import pickle
import numpy as np
import argparse
from TNet import *
from hyperTNet import hyperTN

def identify_time_periods(h_tnet: hyperTN, save_res=True):
    prevalence = np.loadtxt(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'beta1_1.00-beta2_1.00-theta_1.0', 'prevalence2d.txt'), delimiter='\t', dtype=np.float64).mean(axis=1)
    thresholds = np.linspace(0.1, 1, 10) * h_tnet.n
    t_thresholds = -np.ones(len(thresholds), dtype=np.int32)
    i = 0
    for t, p in enumerate(prevalence):
        if p >= thresholds[i]:
            print(t, p)
            t_thresholds[i] = t+1
            i += 1

    if save_res:
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'w') as f:
            f.write(json.dumps(t_thresholds[t_thresholds>0].tolist()))
    return t_thresholds[t_thresholds>0]

def spread_one_pass(h_tnet: hyperTN, shuffled_r, params, rid=0, model='threshold_model'):
    res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, model,
                         'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(params['beta1'], params['beta2'],
                                                                          params['theta']))
    if not path.exists(res_path):
        os.mkdir(res_path)
    prevalence = np.zeros((len(h_tnet.hypercontacts), h_tnet.n), dtype=np.float64)
    backbones = Counter()
    for node in range(h_tnet.n):
        print('Seeds: ', node)
        diffusion_tree_links, prev = h_tnet.threshold_model(seedset=frozenset({node}), params=params, T=h_tnet.T)
        prevalence[:, node] = prev
        backbones.update(diffusion_tree_links)

    if params['beta1'] < 1.0 or params['beta2'] < 1.0:
        suffix = '-r{0}'.format(rid)
    else:
        suffix = ''
    np.savetxt(path.join(res_path, 'prevalence2d{0}{1}.txt'.format(suffix, f'-s{shuffled_r}' if shuffled_r > 0 else '')), prevalence, fmt='%6.1f', delimiter='\t')
    with open(path.join(res_path, 'backbone{0}{1}.pkl'.format(suffix, f'-s{shuffled_r}' if shuffled_r > 0 else '')), 'wb') as f:
        pickle.dump(backbones, f)

def spread_all_subnets(h_tnet: hyperTN, shuffled_r, params, rid=0, model='threshold_model'):  # TODO: maybe accelarate
    res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, model,
                         'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(params['beta1'], params['beta2'],
                                                                          params['theta']))
    if not path.exists(res_path):
        os.mkdir(res_path)
    if params['beta1'] <= 1.0 or params['beta2'] <= 1.0:
        suffix = '-r{0}'.format(rid)
    else:
        suffix = ''
    Ts = identify_time_periods(h_tnet, save_res=False)
    print(h_tnet.dataname)
    for i, T in enumerate(Ts):
        print(i, T)
        prevalence = np.zeros((T, h_tnet.n), dtype=np.float64)
        backbones = Counter()
        for node in range(h_tnet.n):
            diffusion_tree_links, prev = h_tnet.threshold_model(seedset=frozenset({node}), params=params, T=T)
            prevalence[:, node] = prev
            backbones.update(diffusion_tree_links)
        if shuffled_r <= 1:
            np.savetxt(path.join(res_path, 'T_0.{0}-prevalence2d{1}.txt'.format(i+1, suffix)), prevalence, fmt='%6.1f', delimiter='\t')
        with open(path.join(res_path, 'T_0.{0}-backbone{1}{2}.pkl'.format(i+1, suffix, f'-s{shuffled_r}' if shuffled_r > 0 else '')), 'wb') as f:
            pickle.dump(backbones, f)

def parallel_run(h_tnet: hyperTN, model, params, n_tasks_per_array, array_id, sid, shuffled_r):
    res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, model, 'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(params['beta1'], params['beta2'], params['theta']))
    if not path.exists(res_path):
        try:
            os.mkdir(res_path)
        except:
            print('Path already exists. ({0})'.format(path.exists(res_path)))

    Parallel(n_jobs=n_tasks_per_array, backend='loky')(delayed(spread_all_subnets)(h_tnet, shuffled_r, params, r, model) for r in range((array_id-1)*n_tasks_per_array+sid, array_id*n_tasks_per_array+sid))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contation process on temporal higher-order networks')
    parser.add_argument('--dataset', type=str, default='infectious', help='dataset')
    parser.add_argument('--beta1', type=float, default=0.25, help='infectivity for pairwise interaction')
    parser.add_argument('--beta2', type=float, default=1.0, help='infectivity for hyperlink interaction')
    parser.add_argument('--theta', type=float, default=2, help='threshold for contagion to occur in hyperlink interaction')
    parser.add_argument('--R', type=int, default=100, help='the number of realizations in total')
    parser.add_argument('--sid', type=int, default=1, help='the starting id of the realization')
    parser.add_argument('--n_arrays', type=int, default=10, help='the number of job arrays in slurm')
    parser.add_argument('--array_id', type=int, default=1, help='ID of job arrays in slurm')
    parser.add_argument('--shuffled_r', type=int, default=0, help='ID of shuffled network')
    args = parser.parse_args()
    # tnet = TN(args.dataset)
    h_tnet = hyperTN(args.dataset)

    # identify_time_periods(h_tnet)
    # spread_one_pass(h_tnet, args.shuffled_r, {'beta1': args.beta1, 'beta2': args.beta2, 'theta': args.theta})
    if args.beta1 <= 1.0 or args.beta2 <= 1.0:
        parallel_run(h_tnet, 'threshold_model', {'beta1': args.beta1, 'beta2': args.beta2, 'theta': args.theta}, args.R//args.n_arrays, args.array_id, args.sid, args.shuffled_r)
    else:
        spread_all_subnets(h_tnet, args.shuffled_r, {'beta1': args.beta1, 'beta2': args.beta2, 'theta': args.theta})