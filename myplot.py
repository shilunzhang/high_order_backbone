from os import path
from functools import reduce
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from TNet import TN, PATH_TO_RESULTS
from hyperTNet import hyperTN

def line_shaded(x, y, y_diff, ax, line_params=None):
    if line_params is None:
        line_params = dict()
    line = ax.plot(x, y, '-', linewidth=2, clip_on=False, **line_params)
    ax.fill_between(x, y + y_diff, y - y_diff, color='None', facecolor=line[0].get_color(), alpha=0.1)

    return line

def groupsize_statistics(h_tnet: hyperTN, plot=True):
    groupsizes = list(reduce(lambda a,b: a+b, [[len(g) for g in h] for h in h_tnet.hypercontacts]))
    gs, freq = np.unique(groupsizes, return_counts=True)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.8))
        ax.bar(gs, freq)
        ax.set_yscale('log')
        ax.set_xlabel('s')
        ax.set_ylabel('P(s)')
        ax.set_title(h_tnet.tnet.dataname)
        fig.tight_layout()
        fig.savefig(path.join('results', h_tnet.tnet.dataname, 'groupsize_statistics.pdf'), dpi=150)
        plt.show()

    return gs, freq



def plot_prevalence(h_tnet: hyperTN, params):
    res_path = path.join(PATH_TO_RESULTS, h_tnet.tnet.dataname, 'threshold_model', 'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:d}'.format(params['beta1'], params['beta2'], params['theta']))
    prevalence = np.hstack((np.load(path.join(res_path, 'prevalence2d-r{0}.npy'.format(i))).mean(axis=1).reshape(-1, 1) for i in range(1, 101)))
    print(prevalence.shape)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    line_shaded(range(prevalence.shape[0]), np.mean(prevalence, axis=1), y_diff=np.std(prevalence, axis=1), ax=ax,
                line_params={'c': 'g', 'alpha': 0.5})
    # line_shaded(range(prevalence2.shape[0]), np.mean(prevalence2, axis=1), y_diff=np.std(prevalence2, axis=1), ax=ax,
    #             line_params={'c': 'r', 'alpha': 0.5, 'label': r'$\Theta=2$'})
    # line_shaded(range(prevalence3.shape[0]), np.mean(prevalence3, axis=1), y_diff=np.std(prevalence3, axis=1), ax=ax,
    #             line_params={'c': 'b', 'alpha': 0.5, 'label': r'$\Theta=3$'})
    # plt.ylim([0, 1])
    ax.set_xlabel('T')
    ax.set_ylabel(r'$\rho$')
    ax.set_title(h_tnet.tnet.dataname)
    # ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.join('results', h_tnet.tnet.dataname, 'threshold_model',  'prevalence.pdf'), dpi=180)
    plt.show()

# TODO:
def integrate_backbones(dname, theta=2):
    from collections import Counter
    import pickle
    import json
    with open(path.join(PATH_TO_RESULTS, dname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    for i, t in enumerate(ts):
        for beta1 in [0.25]:
            print(i, beta1)
            for beta2 in [1.0]:
                res_path = path.join(PATH_TO_RESULTS, dname, 'threshold_model', 'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:d}'.format(beta1, beta2, theta))
                backbone = Counter()
                for r in range(1, 101):
                    with open(path.join(res_path, 'T_0.{0}-backbone-r{1}.pkl'.format(i+1, r)), 'rb') as f:
                        bb = pickle.load(f)
                        backbone.update(bb)
                        # print(len(bb), len(backbone))
                # return backbone: normalize the weights.
                with open(path.join(res_path, 'T_0.{0}-backbone.pkl'.format(i + 1)), 'wb') as f:
                    pickle.dump(backbone, f)

# beta1 == 1, compare
def backbone_vs_substrate(h_tnet: hyperTN, theta, ax, marker):
    import json, pickle
    global backbone, substrate, ts
    with open(path.join(PATH_TO_RESULTS, h_tnet.tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    n_links = []
    for i in range(1, len(ts)+1):
        print(i, ts[i-1])
        with open(path.join(PATH_TO_RESULTS, h_tnet.tnet.dataname, 'threshold_model', 'beta1_1.00-beta2_1.00-theta_{0}'.format(theta), 'T_0.{0}-backbone.pkl'.format(i)), 'rb') as f:
            backbone = pickle.load(f)
        substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i-1]])
        # print(len(backbone), len(substrate))
        n_links_backbone = Counter([len(e) for e in backbone.keys()])
        n_links_substrate = Counter([len(e) for e in substrate.keys()])
        n_links.append((n_links_backbone, n_links_substrate))
        # print(n_links[-1])

    # max_order = max(n_links[-1][1].keys())
    for ord in range(2, 6):
        ax[ord-2].plot([e[1][ord] for e in n_links], [e[0][ord] for e in n_links], marker=marker, markersize=5, linestyle='None', alpha=0.5, label=h_tnet.tnet.dataname, clip_on=False)
        ax[ord-2].set_aspect('equal')



def all_datasets_backbone_vs_substrate(nets, theta):
    markers = ['o', 'v', '^', '>', '<', 's', 'p', 'P', '*', 'h', 'X', 'D']
    fig, ax = plt.subplots(1, 4, figsize=(9, 2.8))
    for i, dname in enumerate(nets):
        tnet = TN(dname)
        h_tnet = hyperTN(tnet)
        backbone_vs_substrate(h_tnet, theta, ax, marker=markers[i])

    for i, axes in enumerate(ax):
        print(i, axes)
        axes.set_title('order: {0}'.format(i + 2))
        axes.set_clip_on(False)
    ax[0].plot([0, 6000], [0, 6000], '--', c='g', alpha=0.5)
    ax[1].plot([0, 2000], [0, 2000], '--', c='g', alpha=0.5)
    ax[2].plot([0, 150], [0, 150], '--', c='g', alpha=0.5)
    ax[3].plot([0, 10], [0, 10], '--', c='g', alpha=0.5)
    ax[0].set_xlim([0, 6000])
    ax[0].set_ylim([0, 6000])
    ax[1].set_xlim([0, 2000])
    ax[1].set_ylim([0, 2000])
    ax[2].set_xlim([0, 150])
    ax[2].set_ylim([0, 150])
    ax[3].set_xlim([0, 10])
    ax[3].set_ylim([0, 10])
    ax[0].set_xlabel(r'$|\mathcal{L}(G_w)|$')
    ax[1].set_xlabel(r'$|\mathcal{L}(G_w)|$')
    ax[2].set_xlabel(r'$|\mathcal{L}(G_w)|$')
    ax[3].set_xlabel(r'$|\mathcal{L}(G_w)|$')
    ax[0].set_ylabel(r'$|\mathcal{{L}}(G(\beta_1=1, \beta_2=1, \Theta={0}))|$'.format(theta))
    ax[-1].legend(frameon=False, fontsize='small')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, 'figs', 'N_links-theta_{0}.pdf'.format(theta)), dpi=180)

if __name__ == '__main__':
    dname = 'infectious'
    tnet = TN(dname)
    h_tnet = hyperTN(tnet)
    # plot_prevalence(h_tnet, {'beta1': 0.5, 'beta2': 1.0, 'theta':2})
    # integrate_backbones(dname)
    # backbone_vs_substrate(h_tnet)
    all_datasets_backbone_vs_substrate(['highschool2013', 'primaryschool', 'ht09', 'infectious'], 1)