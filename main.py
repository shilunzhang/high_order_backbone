import os
from collections import Counter
from joblib import Parallel, delayed
import pickle
import argparse
import numpy as np
import config
from config import PATH_TO_RESULTS
from hyperTNet import HyperTN


def backbone_many_betas_multiple_runs(h_tnet: HyperTN, theta, beta_list):
    T = 1421  # for subnet T_0.9

    prevalence_diff_beta = []
    for beta in beta_list:
        params = {'beta': beta, 'theta': theta}
        # backbone = Counter()
        prevalence = np.zeros((T, h_tnet.n), dtype=np.float64)
        for node in range(h_tnet.n):
            diffusion_tree_links, prev = h_tnet.simulate_threshold_model(seedset=frozenset({node}), params=params, T=T)
            # backbone.update(diffusion_tree_links)
            prevalence[:, node] = prev
            
        # backbone_list.append(backbone)
        prevalence_diff_beta.append(prevalence.mean(axis=1, keepdims=True))

    return np.hstack(prevalence_diff_beta)

def parallel_run_many_betas(h_tnet, theta, beta_list, n_tasks_per_array, array_id):
    prevalence_list = Parallel(n_jobs=n_tasks_per_array, backend='loky')(delayed(backbone_many_betas_multiple_runs)(h_tnet, theta, beta_list) for r in range((array_id-1)*n_tasks_per_array+1, array_id*n_tasks_per_array+1))

    for idx, beta in enumerate(beta_list):
        res_path = os.path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'beta_{0:.3f}-theta_{1}'.format(beta, theta))
        os.makedirs(res_path, exist_ok=True)

        np.save(os.path.join(res_path, f'T_0.9-prevalence-R{array_id}.npy'), np.mean(np.vstack([prev_diff_beta[:, idx] for prev_diff_beta in prevalence_list]), axis=0))

def average_prevalence(h_tnet: HyperTN, beta_list, theta):
    for beta in beta_list:
        res_path_beta = os.path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'beta_{0:.3f}-theta_{1}'.format(beta, theta))

        if os.path.exists(os.path.join(res_path_beta, 'T_0.9-prevalence-1000.npy')):
            return 0
        else:
            prevalence = np.load(os.path.join(res_path_beta, 'T_0.9-prevalence-R1.npy'))
            for i in range(2, 100):
                prev = np.load(os.path.join(res_path_beta, f'T_0.9-prevalence-R{i}.npy'))
                prevalence = prevalence + prev

            prevalence = prevalence / 100
            np.save(os.path.join(res_path_beta, 'T_0.9-prevalence-1000.npy'), prevalence)
            return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contation process on temporal higher-order networks')
    parser.add_argument('--dataset', type=str, default='SFHH', help='dataset')
    parser.add_argument('--theta', type=str, default='d-1', help='threshold for contagion to occur in hyperlink interaction')
    parser.add_argument('--R', type=int, default=10, help='the number of realizations in total')
    parser.add_argument('--n_arrays', type=int, default=1, help='the number of job arrays in slurm')
    parser.add_argument('--array_id', type=int, default=1, help='ID of job arrays in slurm')
    args = parser.parse_args()

    h_tnet = HyperTN(args.dataset)

    beta_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parallel_run_many_betas(h_tnet, args.theta, beta_list, n_tasks_per_array=args.R//args.n_arrays, array_id=args.array_id)
    # average_prevalence(h_tnet, beta_list, args.theta)  