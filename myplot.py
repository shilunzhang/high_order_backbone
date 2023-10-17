import os.path
import pickle
from os import path
import json
import argparse
from functools import reduce
from collections import Counter
import numpy as np
from scipy.stats import rankdata, pearsonr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from TNet import TN, PATH_TO_RESULTS
from hyperTNet import hyperTN
from centrality import *

MARKERS = ['o', 'v', '^', '>', '<', 's', 'p', 'P', '*', 'h', 'X', 'D']
COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


def line_shaded(x, y, y_diff, ax, line_params=None):
    if line_params is None:
        line_params = dict()
    line = ax.plot(x, y, '-', linewidth=2, clip_on=False, **line_params)
    ax.fill_between(x, y + y_diff, y - y_diff, color='None', facecolor=line[0].get_color(), alpha=0.1)

    return line


def groupsize_statistics(h_tnet: hyperTN, plot=True):
    groupsizes = list(reduce(lambda a, b: a + b, [[len(g) for g in h] for h in h_tnet.hypercontacts]))
    gs, freq = np.unique(groupsizes, return_counts=True)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.8))
        ax.plot(gs, freq, '-o')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('s')
        ax.set_ylabel('f(s)')
        ax.set_title(h_tnet.dataname)
        fig.tight_layout()
        fig.savefig(path.join('results', h_tnet.dataname, 'groupsize_statistics.pdf'), dpi=150)
        plt.show()

    return gs, freq


def plot_prevalence(h_tnet: hyperTN):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model')
    fig, ax = plt.subplots(1, len(beta), figsize=(2.2 * len(beta), 2.2))
    for i, bt in enumerate(beta):
        prevalence1 = np.load(path.join(res_path, 'beta1_{0:.2f}-beta2_{0:.2f}-theta_{1:.1f}'.format(bt, 1), 'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
        prevalence2 = np.load(path.join(res_path, 'beta1_{0:.2f}-beta2_{0:.2f}-theta_{1:.1f}'.format(bt, -1), 'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
        ax[i].plot(range(len(prevalence1)), prevalence1/h_tnet.n, c='black', clip_on=False, alpha=0.6)
        ax[i].plot(range(len(prevalence2)), prevalence2/h_tnet.n, c='maroon', clip_on=False, alpha=0.6)
        ax[i].plot([len(prevalence1)-1], [prevalence1[-1]/h_tnet.n], 'o', c='k', markersize=6, markeredgewidth=0, alpha=0.6, clip_on=False, label=r'${0:.2f}/{1:d}$'.format(prevalence1[-1], h_tnet.n))
        ax[i].plot([len(prevalence2)-1], [prevalence2[-1]/h_tnet.n], 'o', c='maroon', markersize=6, markeredgewidth=0, alpha=0.6, clip_on=False, label=r'${0:.2f}/{1:d}$'.format(prevalence2[-1], h_tnet.n))
        ax[i].xaxis.set_tick_params(bottom=False)
        ax[i].yaxis.set_ticks([0.1 * t for t in range(len(ts)+1)])
        ax[i].yaxis.set_tick_params(direction='in')
        ax[i].legend(frameon=False)
        # ax[i].set_yscale('log')
        ax[i].set_ylim([0, 0.1 * len(ts)])
    ax[0].set_xlabel('')
    ax[0].set_ylabel(r'prevalence')
    # ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.join(res_path, 'prevalence_evolution.pdf'), dpi=200)
    # plt.show()


def integrate_backbones(h_tnet: hyperTN, beta, theta=1, num_r=100):
    if h_tnet.datatype == 'phy-contact':
        num_r = 1000
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    for i, t in enumerate(ts):
        for beta1 in [beta]:
            print(i, beta1)
            for beta2 in [1.0]:
                res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                                     'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(beta1, beta1, theta))
                # if path.exists(path.join(res_path, 'T_0.{0}-backbone.pkl'.format(i + 1))):
                #     print('Existed..')
                #     continue
                backbone = Counter()
                prevalence = np.zeros((t, num_r), dtype=float)
                if os.path.exists(path.join(res_path, 'T_0.{0}-prevalence_averaged.npy'.format(i + 1))):
                    continue
                for r in range(1, num_r + 1):
                    print(r)
                    # with open(path.join(res_path, 'T_0.{0}-backbone-r{1}.pkl'.format(i + 1, r)), 'rb') as f:
                    #     bb = pickle.load(f)
                    #     backbone.update(bb)
                    prevalence[:, r - 1] = np.loadtxt(
                        path.join(res_path, 'T_0.{0}-prevalence2d-r{1}.txt'.format(i + 1, r)), dtype=float).mean(
                        axis=1)
                # return backbone: normalize the weights.
                prevalence_average = prevalence.mean(axis=1)
                # with open(path.join(res_path, 'T_0.{0}-backbone.pkl'.format(i + 1)), 'wb') as f:
                #     pickle.dump(backbone, f)
                np.save(path.join(res_path, 'T_0.{0}-prevalence_averaged.npy'.format(i + 1)), prevalence_average)


# beta1 == 1, compare
def backbone_vs_substrate(h_tnet: hyperTN, theta, ax, marker):
    import json, pickle
    global backbone, substrate, ts
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    n_links = []
    for i in range(1, len(ts) + 1):
        print(i, ts[i - 1])
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                            'beta1_1.00-beta2_1.00-theta_{0:.1f}'.format(theta), 'T_0.{0}-backbone.pkl'.format(i)),
                  'rb') as f:
            backbone = pickle.load(f)
        substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i - 1]])

        n_links_backbone = Counter([len(e) for e in backbone.keys()])
        n_links_substrate = Counter([len(e) for e in substrate.keys()])
        n_links.append((n_links_backbone, n_links_substrate))
        # print(n_links[-1])

    # max_order = max(n_links[-1][1].keys())
    for ord in range(2, 6):
        ax[ord - 2].plot([e[1][ord] for e in n_links], [e[0][ord] for e in n_links], marker=marker, markersize=5,
                         linestyle='None', alpha=0.5, label=h_tnet.dataname, clip_on=False)
        ax[ord - 2].set_aspect('equal')


def all_datasets_backbone_vs_substrate(nets, theta):
    markers = MARKERS
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
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
    fig.savefig(path.join(PATH_TO_RESULTS, 'figs', 'N_links-theta_{0:.1f}.pdf'.format(theta)), dpi=180)


def weights_distance(substrate, backbone):
    err = 0
    for k in substrate:
        if k in backbone:
            err += np.abs(substrate[k] - backbone[k])
        else:
            err += substrate[k]
    return err / len(substrate)


def weights_distance_normalized(substrate, backbone):
    err = 0
    for k in substrate:
        if k in backbone:
            err += np.abs(substrate[k] - backbone[k]) / substrate[k]
        else:
            err += 1
    return err


def compare_backbones_substrate(h_tnet: hyperTN, ax, theta=1):
    ''' Recall, i.e., the fraction of hlinks that appear in the backbone, as a function of beta.
    :param h_tnet: the dataset
    :param q: the number of links considered
    :param which: which backbone to compare with, beta==0 or 1
    :return:
    '''
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    for i in range(len(ts)):
        substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
        backbone1 = substrate

        r_beta = []
        for b1 in beta:
            print(i, b1)
            with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                                'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(b1, b1, theta),
                                'T_0.{0}-backbone.pkl'.format(i + 1)), 'rb') as f:
                backbone2 = pickle.load(f)
            r_beta.append(len(backbone1 & backbone2) / len(backbone1))
        ax.plot(beta, r_beta, marker=MARKERS[i], label='${0}0\\%\\cdot T$'.format(i + 1), clip_on=False, alpha=0.5)


def compare_backbones_substrate_fix_order(h_tnet: hyperTN, ax, order, theta=1):
    ''' Recall, i.e., the fraction of hlinks that appear in the backbone, as a function of beta.
    :param h_tnet: the dataset
    :param q: the number of links considered
    :param which: which backbone to compare with, beta==0 or 1
    :return:
    '''
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    for i in range(len(ts)):
        substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
        backbone1 = substrate

        r_beta = []
        for b1 in beta:
            print(i, b1)
            with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                                'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(b1, b1, theta),
                                'T_0.{0}-backbone.pkl'.format(i + 1)), 'rb') as f:
                backbone2 = pickle.load(f)
            r_beta.append(
                len({k for k in backbone1 if len(k) == order} & {k for k in backbone2 if len(k) == order}) / len(
                    {k for k in backbone1 if len(k) == order}))
        ax.plot(beta, r_beta, color=COLORS[order - 2], marker=MARKERS[i], label='${0}0\\%\\cdot T$'.format(i + 1),
                clip_on=False, alpha=0.5)


def plot_backbone_comparison(h_tnet: hyperTN, order, theta=1):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    suffix = ''
    if order > 1:
        compare_backbones_substrate_fix_order(h_tnet, ax=ax, order=order, theta=theta)
        suffix = '-order_{}'.format(order)
    else:
        compare_backbones_substrate(h_tnet, ax=ax, theta=theta)
    # compare_backbones_substrate_weights(h_tnet, ax=ax, theta=theta)
    # scatter_backbones_substrate_weights(h_tnet, ax=ax, theta=theta)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0., 1.0])
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Recall')
    # ax.set_ylabel(r'$r(G(\beta_1=\beta, \beta_2=\beta, \Theta=1), G_w)$')
    # ax.set_ylabel(r'Distance$(G(\beta_1=\beta, \beta_2=\beta, \Theta=1), G_w^*)$')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'Recall-theta_{0:.1f}{1}.pdf'.format(theta, suffix)), dpi=180)
    # fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'Weights_scatter-theta_{0}.pdf'.format(theta)), dpi=180)


def compare_diff_threshold_beta_1(h_tnet: hyperTN):
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    fig, ax = plt.subplots(1, len(ts), figsize=(2.2 * len(ts), 2.2), sharey=True)
    for i in range(len(ts)):
        substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
        max_order = max({len(hlink) for hlink in substrate})
        data1, data2 = [], []
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                            'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(1.0, 1.0, 1),
                            'T_0.{0}-backbone.pkl'.format(i + 1)), 'rb') as f:
            backbone1 = pickle.load(f)
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                            'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(1.0, 1.0, -1),
                            'T_0.{0}-backbone.pkl'.format(i + 1)), 'rb') as f:
            backbone2 = pickle.load(f)
        for k in range(2, max_order + 1):
            num_links1 = len({hlink for hlink in backbone1 if len(hlink) == k})
            num_links2 = len({hlink for hlink in backbone2 if len(hlink) == k})
            if num_links1 > 0:
                data1.append((k, num_links1))
            if num_links2 > 0:
                data2.append((k, num_links2))
        ax[i].plot([e[0] for e in data1], [e[1] for e in data1], '-', linewidth=0.7, markersize=2.5, marker=MARKERS[0],
                   alpha=0.6, label='$\Theta=1$')
        ax[i].plot([e[0] for e in data2], [e[1] for e in data2], '-', linewidth=0.7, markersize=2.5, marker=MARKERS[1],
                   alpha=0.6, label='$\Theta=h-1$')
        ax[i].set_xticks(list(range(1, max_order + 2)))
        ax[i].set_xlim([1, max_order + 1])
        ax[i].set_xlabel('order')

        ax[i].set_title('$T_{{{0}\\%}}$'.format((i + 1) * 10))
        # ax[i].set_yscale('log')
    ax[0].set_ylabel('#hyperlinks')
    ax[0].legend(frameon=False, loc=3, ncol=1, borderpad=0, labelspacing=0.1)
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'Number_links_beta_1_linear.pdf'),
                dpi=180)


def scatter_backbones_substrate_weights(h_tnet: hyperTN, its, beta, ax, theta=1, ranking=False):
    '''
    :param h_tnet: the dataset
    :param q: the number of links considered
    :param which: which backbone to compare with, beta==0 or 1
    :return:
    '''

    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))

    substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    substrate_rescaled = dict()
    if theta == -1:
        substrate_rescaled = {k: v * len(k) * beta for k, v in dict(substrate).items()}
    elif theta == 1:
        substrate_rescaled = {k: v * len(k) * (len(k) - 1) * beta for k, v in dict(substrate).items()}
    elif 0 < theta < 1:
        substrate_rescaled = {k: v * len(k) * (len(k) - np.ceil(theta * len(k))) * beta for k, v in
                              dict(substrate).items()}

    r_beta = []
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                        'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(beta, beta, theta),
                        'T_0.{0}-backbone.pkl'.format(its + 1)), 'rb') as f:
        backbone = pickle.load(f)
    backbone = {k: v / 1000 for k, v in backbone.items()}
    # backbone_normalized = {k: v/(len(k)*(len(k)-1)*beta) for k, v in dict(backbone).items()}
    for k in substrate_rescaled:
        if k in backbone:
            r_beta.append((substrate_rescaled[k], backbone[k], len(k)))
        else:
            r_beta.append((substrate_rescaled[k], 0, len(k)))
    if ranking:
        ranking1 = 1 + len(r_beta) - rankdata([e[0] for e in r_beta], method='average')
        ranking2 = 1 + len(r_beta) - rankdata([e[1] for e in r_beta], method='average')
        scatters = []
        for i in range(len(r_beta)):
            scatters.append((ranking1[i], ranking2[i], r_beta[i][2]))
        ax.scatter([e[0] for e in scatters], [e[1] for e in scatters], marker='o',
                   c=[COLORS[e[2] - 2] for e in scatters],
                   s=[1.5 + (e[2] - 2) ** 1.5 * 4 for e in scatters], clip_on=False, alpha=0.5)
        ax.set_xlim([1, len(scatters) + 1])
        ax.set_ylim([1, len(scatters) + 1])
        # ax.axis('equal')
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        xmin = min([min([e[0] for e in r_beta]), min([e[1] for e in r_beta])])
        xmax = max([max([e[0] for e in r_beta]), max([e[1] for e in r_beta])])
        print('xmin: ', xmin)
        print('xmax: ', xmax)
        ax.plot([xmin, xmax], [xmin, xmax], '--', linewidth=0.6, color='grey', alpha=0.7)
        ax.scatter([e[0] for e in r_beta], [e[1] for e in r_beta], marker='o', c=[COLORS[e[2] - 2] for e in r_beta],
                   s=[1.5 + (e[2] - 2) ** 1.5 * 4 for e in r_beta], clip_on=False, alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([xmin, xmax])


def scatter_weights4subnets(h_tnet: hyperTN, theta=1, ranking=False):
    suffix = 'rank' if ranking else ''
    # beta = [0.01, 0.05, 0.1, 0.5, 1.0]
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    fig, ax = plt.subplots(len(ts), len(beta), figsize=(1.7 * len(beta), 1.7 * len(ts)))
    for i, t in enumerate(ts):
        for j, b in enumerate(beta):
            # ax[i][j].axis('equal')
            scatter_backbones_substrate_weights(h_tnet, its=i, beta=b, ax=ax[i][j], theta=theta, ranking=ranking)
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'Weights_scatter_all-theta_{0:.1f}_{1}_.pdf'.format(theta, suffix)), dpi=180)


def plot_distance_weights(h_tnet: hyperTN, ax, theta):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    for its, t in enumerate(ts):
        substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
        substrate_rescaled = {k: v * len(k) for k, v in dict(substrate).items()}
        dist = []
        for b in beta:
            if theta == -1:
                substrate_rescaled = {k: v * b for k, v in dict(substrate).items()}
            elif theta == 1:
                substrate_rescaled = {k: v * (len(k) - 1) * b for k, v in dict(substrate).items()}
            elif 0 < theta < 1:
                substrate_rescaled = {k: v * (len(k) - np.ceil(theta * len(k))) * b for k, v in dict(substrate).items()}
            with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                                'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(b, b, theta),
                                'T_0.{0}-backbone.pkl'.format(its + 1)), 'rb') as f:
                backbone = pickle.load(f)
            backbone = {k: v / 1000 for k, v in backbone.items()}
            # backbone_normalized = {k: v / (len(k)*(len(k)-1)*b) for k, v in dict(backbone).items()}
            dist.append(weights_distance_normalized(substrate_rescaled, backbone))
        ax.plot(beta, dist, marker=MARKERS[its], markersize=6, markerfacecolor='white', markeredgewidth=1.1,
                label='${0}0\\%\\cdot T$'.format(its + 1))


def distance_weights4subnets(theta=1):
    '''
    (Normalizaed) Distance between weighted substrate network and the diffusion network as a function of infection rate
    :param theta:
    :return:
    '''
    datasets = ['infectious', 'ht09', 'highschool2013', 'primaryschool']
    fig, ax = plt.subplots(1, len(datasets), figsize=(3 * len(datasets), 3))
    for d, dname in enumerate(datasets):
        tnet = TN(dname)
        h_tnet = hyperTN(tnet)
        plot_distance_weights(h_tnet, ax=ax[d], theta=theta)
        ax[d].set_xlabel(r'$\beta$')
        ax[d].set_title(dname)
    ax[0].set_ylabel('weight distance')
    ax[-1].legend(frameon=False, labelspacing=0)
    plt.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, 'figs',
                          'Normalized_Distance_weights-theta_{0:.1f}__.pdf'.format(theta)), dpi=180)


def scatter_backbones_weights(h_tnet: hyperTN, its, beta, theta, ax, order, normalization=False, ranking=False):
    r_beta = []
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                        'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(beta, beta, 1),
                        'T_0.{0}-backbone.pkl'.format(its + 1)), 'rb') as f:
        backbone1 = pickle.load(f)
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                        'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(beta, beta, theta),
                        'T_0.{0}-backbone.pkl'.format(its + 1)), 'rb') as f:
        backbone2 = pickle.load(f)
    n_realizations = 1000
    backbone1 = {k: v / n_realizations for k, v in backbone1.items()}
    backbone2 = {k: v / n_realizations for k, v in backbone2.items()}
    hlinks = set().union(backbone1.keys(), backbone2.keys())
    for k in hlinks:
        if k in backbone1 or k in backbone2:
            r_beta.append((backbone1[k] if k in backbone1 else 0, backbone2[k] if k in backbone2 else 0, len(k)))
    if ranking:
        ranking1 = 1 + len(r_beta) - rankdata([e[0] for e in r_beta], method='average')
        ranking2 = 1 + len(r_beta) - rankdata([e[1] for e in r_beta], method='average')
        scatters = []
        for i in range(len(r_beta)):
            scatters.append((ranking1[i], ranking2[i], r_beta[i][2]))
        ax.scatter([e[0] for e in scatters], [e[1] for e in scatters], marker='o',
                   c=[COLORS[e[2] - 2] for e in scatters],
                   s=[1.5 + (e[2] - 2) ** 1.5 * 4 for e in scatters], clip_on=False, alpha=0.5)
        ax.set_xlim([1, len(scatters) + 1])
        ax.set_ylim([1, len(scatters) + 1])
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        if normalization:
            print('Weight with normalization')
            r_beta = [(e[0]/(e[2]-1), e[1]/(1 if theta == -1 else e[2] - np.ceil(theta * e[2])), e[2]) for e in r_beta]
        if order > 0:
            r_beta = [e for e in r_beta if e[2] == order]
        print(r_beta[0])
        xmin = min([min([e[0] for e in r_beta]), min([e[1] for e in r_beta])])
        xmax = max([max([e[0] for e in r_beta]), max([e[1] for e in r_beta])])
        ax.plot([xmin, xmax], [xmin, xmax], '--', linewidth=0.6, color=COLORS[0], alpha=0.5)
        ax.scatter([e[0] for e in r_beta], [e[1] for e in r_beta], marker='o', c=[COLORS[e[2]-2] for e in r_beta], s=[1.5 + (e[2]-2) ** 1.5 * 4 for e in r_beta], clip_on=False, alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([xmin, xmax])


def scatter_weights_diff_thresholds(h_tnet: hyperTN, theta, order, normalization, ranking=False):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    fig, ax = plt.subplots(len(ts), len(beta), figsize=(1.7 * len(beta), 1.7 * len(ts)))
    for i, t in enumerate(ts):
        for j, b in enumerate(beta):
            # ax[i][j].axis('equal')
            scatter_backbones_weights(h_tnet, its=i, beta=b, theta=theta, ax=ax[i][j], order=order, normalization=normalization, ranking=ranking)
    fig.tight_layout()
    suffix = ''
    suffix += '_order{0}'.format(order) if order > 0 else ''
    suffix += '_normalization' if normalization else ''
    suffix += '_rank' if ranking else ''
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'Weights_scatter_diff_thresholds_theta_{0:.1f}{1}.pdf'.format(theta, suffix)), dpi=180)


def activated_links(h_tnet: hyperTN):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    theta = [1, 0.5, -1]


def compare_backbone_diff_t_heatmap(h_tnet: hyperTN, top_n, theta=1):  # TODO: heatmap, for a fixed order
    '''
    Compare importance of links in different sub-temporalnets (corresponding to different periods in the same temporal network).
    measured by 1, overlap in top-N links; 2, ...
    :param h_tnet:
    :param theta:
    :return:
    '''
    beta = [0.01, 0.05, 0.1, 0.5, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts)
    fig, ax = plt.subplots(1, len(beta), figsize=(len(beta) * 2.4, 2.4))
    for i, bt in enumerate(beta):
        normalization = 1000
        data = np.ones((len(ts), len(ts)), dtype=np.float64)
        for j, t in enumerate(ts):
            with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                                'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(bt, bt, theta),
                                'T_0.{0}-backbone.pkl'.format(j + 1)), 'rb') as f:
                backbone1 = pickle.load(f)
            backbone1 = {k: v / (normalization * len(k) * (len(k) - 1) * bt) for k, v in dict(backbone1).items()}
            for k in range(j + 1, len(ts)):
                with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                                    'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(bt, bt, theta),
                                    'T_0.{0}-backbone.pkl'.format(k + 1)), 'rb') as f:
                    backbone2 = pickle.load(f)
                backbone2 = {k: v / (normalization * len(k) * (len(k) - 1) * bt) for k, v in dict(backbone2).items()}
                link_weights1 = [backbone1[k] if k in backbone1 else 0 for k in substrate]
                link_weights2 = [backbone2[k] if k in backbone2 else 0 for k in substrate]
                data[j, k] = overlap_top_N(link_weights1, link_weights2, top_n=top_n)
                data[k, j] = data[j, k]
        sns.heatmap(data[::-1], cmap='Blues', ax=ax[i],
                    xticklabels=['$T_{{{0}\\%}}$'.format(10 * i) for i in range(1, data.shape[0] + 1)],
                    yticklabels=['$T_{{{0}\\%}}$'.format(10 * i) for i in range(1, data.shape[1] + 1)][::-1],
                    annot=False, fmt='.2f', annot_kws={'fontsize': 'x-small'}, vmin=0, vmax=1.0)
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'top_links_diff_t_heatmap_theta_{0:.1f}.pdf'.format(theta)), dpi=180)


def scatter_weights_topo_props(h_tnet: hyperTN, its, metric='weights', order=2):
    global hlinks_order3_metric
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_realizations = 1000
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))

    substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    if metric == 'weights':
        hlinks_order3_metric = effective_weights_order3(h_tnet, inverse=False, i=its)
    elif metric == 'inverse weights':
        hlinks_order3_metric = effective_weights_order3(h_tnet, inverse=True, i=its)
    links_timestamps = dict().fromkeys(substrate)
    for k in substrate:
        links_timestamps[k] = []
    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[its]]):
        for hlink in hlinks:
            links_timestamps[hlink].append(t)

    fig, ax = plt.subplots(2, len(beta), figsize=(2.2 * len(beta), 2.2 * 2))
    for i, bt in enumerate(beta):
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                            'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(bt, bt, 1),
                            'T_0.{0}-backbone.pkl'.format(its + 1)), 'rb') as f:
            backbone1 = pickle.load(f)
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                            'beta1_{0:.2f}-beta2_{1:.2f}-theta_{2:.1f}'.format(bt, bt, -1),
                            'T_0.{0}-backbone.pkl'.format(its + 1)), 'rb') as f:
            backbone2 = pickle.load(f)
        backbone1 = {k: v / n_realizations for k, v in backbone1.items()}
        backbone2 = {k: v / n_realizations for k, v in backbone2.items()}
        data1 = [(np.sum(links_timestamps[k]), backbone1[k], substrate[k], hlinks_order3_metric[k]) for k in backbone1
                 if len(k) == order]
        # data1 = [(np.sum([1/(e+1) for e in links_timestamps[k]]), backbone1[k], substrate[k]) for k in backbone1 if len(k) == order]
        ax[0][i].scatter([e[2] for e in data1], [e[1] for e in data1], c=[np.log(e[2]) for e in data1], cmap='viridis',
                         alpha=0.5)
        r = pearsonr([e[2] for e in data1], [e[1] for e in data1])[0]
        tau = kendalltau([e[2] for e in data1], [e[1] for e in data1])[0]
        ax[0][i].set_title('r={0:.3f}, tau={1:.3f}'.format(r, tau), pad=0.1)
        data2 = [(np.sum(links_timestamps[k]), backbone2[k], substrate[k], hlinks_order3_metric[k]) for k in backbone2
                 if len(k) == order]
        # data2 = [(np.sum([1/(e+1) for e in links_timestamps[k]]), backbone2[k], substrate[k]) for k in backbone2 if len(k) == order]
        ax[1][i].scatter([e[3] for e in data2], [e[1] for e in data2], c=[np.log(e[2]) for e in data2], cmap='viridis',
                         alpha=0.5)
        r = pearsonr([e[3] for e in data2], [e[1] for e in data2])[0]
        tau = kendalltau([e[3] for e in data2], [e[1] for e in data2])[0]
        ax[1][i].set_title('r={0:.3f}, tau={1:.3f}'.format(r, tau), pad=0.1)
        ax[0][i].set_xscale('log')
        ax[0][i].set_yscale('log')
        ax[1][i].set_xscale('log')
        ax[1][i].set_yscale('log')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'weights_topo_props_order_{0}_effectiveweights_inverse.pdf'.format(order)), dpi=200)

def scatter_weights_topo_props_order2(h_tnet: hyperTN, theta, phi):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ts = h_tnet.time_division(which='all')
    its = len(ts) - 1

    substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    # hlinks_order2_2hop_metric = two_hop_score_order2(h_tnet, i=its)

    backbone_beta1 = h_tnet.return_backbone({'beta': 1.0, 'theta': 1}, subnet=its)
    hlinks_order2_metric = dict.fromkeys([k for k in substrate if len(k) == 2], 0)
    for k in backbone_beta1:
        if len(k) == 2:
            hlinks_order2_metric[k] = backbone_beta1[k]

    links_timestamps = dict().fromkeys(substrate)
    for k in substrate:
        links_timestamps[k] = []
    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[its]]):
        for hlink in hlinks:
            links_timestamps[hlink].append(t)

    fig, ax = plt.subplots(2, len(beta), figsize=(2.2 * len(beta), 2.2 * 2))
    for i, bt in enumerate(beta):
        print(i, bt)
        backbone = h_tnet.return_backbone({'beta': bt, 'theta': theta}, subnet=its)
        hlinks_order2_2hop_metric = local_effective_superset_metric(h_tnet, i=its, order=2, beta=bt)

        for k in hlinks_order2_metric:
            # hlinks_order2_metric[k] = substrate[k] + bt * hlinks_order2_2hop_metric[k] / 2 + phi * hlinks_order2_metric[k]
            hlinks_order2_metric[k] = substrate[k] + hlinks_order2_2hop_metric[k] + phi * hlinks_order2_metric[k]
        data1 = [(np.sum(links_timestamps[k]), backbone[k], substrate[k], hlinks_order2_metric[k]) for k in backbone
                 if len(k) == 2]
        ax[0][i].scatter([e[2] for e in data1], [e[1] for e in data1], c=[np.log(e[2]) for e in data1], cmap='viridis',
                         alpha=0.5)
        r = pearsonr([e[2] for e in data1], [e[1] for e in data1])[0]
        tau = kendalltau([e[2] for e in data1], [e[1] for e in data1])[0]
        ax[0][i].set_title('r={0:.3f}, tau={1:.3f}'.format(r, tau), pad=0.1)

        data2 = [(np.sum(links_timestamps[k]), backbone[k], substrate[k], hlinks_order2_metric[k]) for k in backbone
                 if len(k) == 2]
        ax[1][i].scatter([e[3] for e in data2], [e[1] for e in data2], c=[np.log(e[2]) for e in data2], cmap='viridis',
                         alpha=0.5)
        r = pearsonr([e[3] for e in data2], [e[1] for e in data2])[0]
        tau = kendalltau([e[3] for e in data2], [e[1] for e in data2])[0]
        ax[1][i].set_title('r={0:.3f}, tau={1:.3f}'.format(r, tau), pad=0.1)
        ax[0][i].set_xscale('log')
        ax[0][i].set_yscale('log')
        ax[1][i].set_xscale('log')
        ax[1][i].set_yscale('log')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'weights_topo_props_order_2_theta{0}_superset_weight_.pdf'.format(theta)), dpi=200)
    # fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
    #                       'weights_topo_props_order_2_one-two_hop_walks_alpha_{0:.2f}.pdf'.format(alpha)), dpi=200)

def scatter_weights_topo_props_diff_order(h_tnet: hyperTN, theta):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ts = h_tnet.time_division(which='all')
    its = len(ts) - 1

    substrate = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    max_order = min([max([len(k) for k in substrate]), 4])
    # metrics = substrate
    # local_eff_metrics = local_effective_sum_metric(h_tnet)
    local_eff_inverse_metrics = local_effective_sum_inverse_metric(h_tnet)
    twohop_metrics = two_hop_walk_score(h_tnet)
    # metrics = local_effective_prod_metric(h_tnet, i=its)
    # metrics = local_cross_order_weight(h_tnet, i=its, supsub='superset')
    # metrics = local_cross_order_effective_weight(h_tnet, i=its, supsub='superset')

    links_timestamps = dict().fromkeys(substrate)
    for k in substrate:
        links_timestamps[k] = []
    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[its]]):
        for hlink in hlinks:
            links_timestamps[hlink].append(t)

    fig, ax = plt.subplots(max_order-1, len(beta), figsize=(2.2 * len(beta), 2.2 * (max_order-1)))
    for i, bt in enumerate(beta):
        # metrics = {k: substrate[k] + bt * twohop_metrics[k] for k in substrate}
        # metrics = {k: substrate[k] + bt * local_eff_metrics[k] for k in substrate}
        # metrics = {k: substrate[k] + bt * (local_eff_metrics[k] + twohop_metrics[k]) for k in substrate}
        metrics = local_eff_inverse_metrics
        backbone = h_tnet.return_backbone({'beta': bt, 'theta': theta}, subnet=its)
        for order in range(2, max_order+1):
            data = [(np.sum(links_timestamps[k]), backbone[k], substrate[k], metrics[k]) for k in backbone if len(k) == order]
            print('Order ', order, ' #datapoints: ', len(data))
            if len(data) < 5:
                continue
            ax[order - 2][i].scatter([e[3] for e in data], [e[1] for e in data], c=[np.log(e[2]) for e in data],
                                     cmap='viridis', alpha=0.5)
            r = pearsonr([e[3] for e in data], [e[1] for e in data])[0]
            tau = kendalltau([e[3] for e in data], [e[1] for e in data])[0]
            ax[order-2][i].set_title('r={0:.3f}, tau={1:.3f}'.format(r, tau), pad=0.1)
            ax[order-2][i].set_xscale('log')
            ax[order-2][i].set_yscale('log')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'weights_topo_props_diff_order_local_eff_plus1_inverse_only_theta_{0}_subnet{1}.pdf'.format(theta, its)), dpi=200)
    # fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
    #                       'weights_topo_props_diff_order_linkweight_theta_{0}_subnet{1}.pdf.pdf'.format(theta, its)), dpi=200)

def number_of_links_activated(h_tnet: hyperTN, weighted=False):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    beta = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_theta1, y_theta2 = np.zeros((len(beta), 2), dtype=float), np.zeros((len(beta), 2), dtype=float)
    for i, bt in enumerate(beta):
        backbone1 = h_tnet.return_backbone({'beta': bt, 'theta': 1}, subnet=h_tnet.time_division(which='largest')[0])
        backbone2 = h_tnet.return_backbone({'beta': bt, 'theta': -1}, subnet=h_tnet.time_division(which='largest')[0])
        if not weighted:
            y_theta1[i, 0] = len([1 for k in backbone1 if len(k) == 2])
            y_theta1[i, 1] = len([1 for k in backbone1 if len(k) == 3])
            y_theta2[i, 0] = len([1 for k in backbone2 if len(k) == 2])
            y_theta2[i, 1] = len([1 for k in backbone2 if len(k) == 3])
        else:
            y_theta1[i, 0] = np.sum([backbone1[k] for k in backbone1 if len(k) == 2]) / (h_tnet.n**2)
            y_theta1[i, 1] = np.sum([backbone1[k] for k in backbone1 if len(k) == 3]) / (h_tnet.n**2)
            y_theta2[i, 0] = np.sum([backbone2[k] for k in backbone2 if len(k) == 2]) / (h_tnet.n**2)
            y_theta2[i, 1] = np.sum([backbone2[k] for k in backbone2 if len(k) == 3]) / (h_tnet.n**2)
    ax.plot(beta, y_theta1[:, 0], marker=MARKERS[0], markersize=6, c=COLORS[0])
    ax.plot(beta, y_theta1[:, 1], marker=MARKERS[1], markersize=6, c=COLORS[1])
    ax.plot(beta, y_theta2[:, 0], '--', marker=MARKERS[0], markersize=6, c=COLORS[0])
    ax.plot(beta, y_theta2[:, 1], '--', marker=MARKERS[1], markersize=6, c=COLORS[1])
    ax.set_yscale('log')
    ax.set_ylabel('percentage of links' if not weighted else '$\sum_j w_j^B/N^2$')
    ax.set_xlabel(r'$\beta$')
    ax.set_xlim([0, 1.05])
    ax.set_title(h_tnet.dataname)
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'percentage_links_activated_beta.pdf' if not weighted else 'weight_link_beta.pdf'), dpi=200)

def ratio_of_links_activated(h_tnet: hyperTN):
    its, ts = h_tnet.time_division(which='largest')
    aggregated = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts])
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    beta = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_theta1, y_theta2 = np.zeros((len(beta), 2), dtype=float), np.zeros((len(beta), 2), dtype=float)
    for i, bt in enumerate(beta):
        backbone1 = h_tnet.return_backbone({'beta': bt, 'theta': 1}, subnet=its)
        backbone2 = h_tnet.return_backbone({'beta': bt, 'theta': -1}, subnet=its)
        y_theta1[i, 0] = len([1 for k in backbone1 if len(k) == 2]) / len([1 for k in aggregated if len(k) == 2])
        y_theta1[i, 1] = len([1 for k in backbone1 if len(k) == 3]) / len([1 for k in aggregated if len(k) == 3])
        y_theta2[i, 0] = len([1 for k in backbone2 if len(k) == 2]) / len([1 for k in aggregated if len(k) == 2])
        y_theta2[i, 1] = len([1 for k in backbone2 if len(k) == 3]) / len([1 for k in aggregated if len(k) == 3])
    ax.plot(beta, y_theta1[:, 0], marker=MARKERS[0], markersize=6, c=COLORS[0])
    ax.plot(beta, y_theta1[:, 1], marker=MARKERS[1], markersize=6, c=COLORS[1])
    ax.plot(beta, y_theta2[:, 0], '--', marker=MARKERS[0], markersize=6, c=COLORS[0])
    ax.plot(beta, y_theta2[:, 1], '--', marker=MARKERS[1], markersize=6, c=COLORS[1])
    # ax.set_yscale('log')
    ax.set_ylabel('percentage of links')
    ax.set_xlabel(r'$\beta$')
    ax.set_xlim([0, 1.05])
    ax.set_title(h_tnet.dataname)
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'percentage_links_activated_beta.pdf'), dpi=200)

def check_low_weight(h_tnet: hyperTN, beta):
    ts = h_tnet.time_division(which='all')
    its = len(ts) - 1
    aggregated = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    backbone = h_tnet.return_backbone({'beta': beta, 'theta': 1}, subnet=its)

    hlinks_order2 = dict().fromkeys([k for k in aggregated if len(k) == 2 and aggregated[k] == 1], 0)
    hlinks_order2_ts = dict().fromkeys([k for k in aggregated if len(k) == 2 and aggregated[k] == 1])
    hlinks_order2_adjcent = dict().fromkeys([k for k in aggregated if len(k) == 2 and aggregated[k] == 1], 0)
    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[its]]):
        for hlink in hlinks:
            if len(hlink) == 2:
                hlinks_order2_ts[hlink] = t+1
        for hlink in hlinks:
            if len(hlink) > 2:
                for hl in hlinks_order2_ts:
                    if hlink.issuperset(hl) and hlinks_order2_ts[hl] is None:
                        hlinks_order2[hl] += len(hlink)
                    if len(hlink.intersection(hl)) >= 1 and hlinks_order2_ts[hl] is None:
                        hlinks_order2_adjcent[hl] += len(hlink)

    data = [(aggregated[k], backbone[k] if k in backbone else 0, hlinks_order2_adjcent[k]) for k in hlinks_order2]
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.scatter([e[0] for e in data], [e[2]+1 for e in data], c=[np.log2(e[1]+1) for e in data],
                                     cmap='viridis', alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(xmin=1)
    ax.set_xlabel('$w^B_j$')
    ax.set_ylabel('superset size')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'check_2d.pdf'), dpi=200)

def check_2d(h_tnet: hyperTN):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ts = h_tnet.time_division(which='all')
    its = len(ts) - 1
    aggregated = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    hlinks_order2 = dict().fromkeys([k for k in aggregated if len(k) == 2], 0)

    fig, ax = plt.subplots(1, len(beta), figsize=(2.2 * len(beta), 2.2))
    for b, bt in enumerate(beta):
        backbone = h_tnet.return_backbone({'beta': bt, 'theta': 1}, subnet=its)

        hlinks_order2_metric = local_effective_superset_metric(h_tnet, i=its, order=2, beta=bt)

        data = [(aggregated[k], backbone[k] if k in backbone else 0, hlinks_order2_metric[k]) for k in hlinks_order2]
        ax[b].scatter([e[0] for e in data], [e[2] + 1 for e in data], c=[np.log2(e[1] + 1) for e in data],
                   cmap='viridis', alpha=0.5)
        ax[b].set_xscale('log')
        ax[b].set_yscale('log')
        ax[b].set_xlabel('$w_j$')
    ax[0].set_ylabel('superset size')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'check_2d_superset.pdf'), dpi=200)

def corr_coef_backbone_metric(h_tnet: hyperTN, params: dict, subnet: int, order: int, metric: str, corr_metric: str):
    global metric_dict
    if metric == 'linkweight':
        metric_dict = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:h_tnet.time_division(which='all')[subnet]])
    else:
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'centrality', 'T_0.{0}-{1}.pkl'.format(subnet+1, metric)), 'rb') as f:
            metric_dict = pickle.load(f)
    # elif metric == 'local_effective_sum_metric':
    #     metric_dict = local_effective_sum_metric(h_tnet, subnet=subnet)
    # elif metric == 'local_effective_sum_inverse_metric':
    #     metric_dict = local_effective_sum_inverse_metric(h_tnet, subnet=subnet)
    # elif metric == 'two_hop_walk_score':
    #     metric_dict = two_hop_walk_score(h_tnet, subnet=subnet)

    metric_dict = {k:v for k,v in metric_dict.items() if len(k) == order}
    backbone = h_tnet.return_backbone(params, subnet=subnet)
    data = [(backbone[k], metric_dict[k]) if k in backbone else (0, metric_dict[k]) for k in metric_dict]

    if corr_metric == 'kendalltau':
        return kendalltau([e[0] for e in data], [e[1] for e in data])[0]
    elif corr_metric == 'pearsonr':
        return pearsonr([e[0] for e in data], [e[1] for e in data])[0]


def compare_diff_metrics(order=3, theta=-1, corr_metric='kendalltau'):
    dataset_phy_contacts = ['infectious', 'primaryschool', 'highschool2013', 'ht09']
    dataset_sci_collab = ['q-bio', 'q-fin', 'hep-lat', 'nucl-th']
    metrics = ['linkweight', 'local_effective_sum_metric', 'local_effective_sum_inverse_metric', 'two_hop_walk_score']
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, ax = plt.subplots(2, 4, figsize=(len(dataset_phy_contacts) * 3.2, 2 * 2.5))
    plt.subplots_adjust(left=0.06, right=0.89, bottom=0.11, top=0.97)
    for idx in range(ax.shape[1]):
        h_tnet1 = hyperTN(dataset_phy_contacts[idx])
        ts1 = h_tnet1.time_division(which='all')
        corr_coef1 = []
        for bt in beta:
            corr_coef1.append([corr_coef_backbone_metric(h_tnet1, {'beta': bt, 'theta': theta}, subnet=len(ts1)-1, order=order, metric=metric, corr_metric=corr_metric) for metric in metrics])
        for m in range(len(metrics)):
            ax[0][idx].plot(beta, [e[m] for e in corr_coef1], '-', marker=MARKERS[m], c=COLORS[m], label=metrics[m])
        h_tnet2 = hyperTN(dataset_sci_collab[idx])
        ts2 = h_tnet2.time_division(which='all')
        corr_coef2 = []
        for bt in beta:
            corr_coef2.append([corr_coef_backbone_metric(h_tnet2, {'beta': bt, 'theta': theta}, subnet=len(ts2) - 1,
                                order=order, metric=metric, corr_metric=corr_metric) for metric in metrics])
        for m in range(len(metrics)):
            ax[1][idx].plot(beta, [e[m] for e in corr_coef2], '-', marker=MARKERS[m], c=COLORS[m], label=metrics[m])
        ax[1][idx].set_xlabel(r'$\beta$')
        plt.text(.97, .97, dataset_phy_contacts[idx], ha='right', va='top', transform=ax[0][idx].transAxes, fontsize='large', fontweight='bold')
        plt.text(.97, .97, dataset_sci_collab[idx], ha='right', va='top', transform=ax[1][idx].transAxes, fontsize='large', fontweight='bold')
    ax[0][0].set_ylabel('Kendall tau')
    ax[1][0].set_ylabel('Kendall tau')
    handles, labels = ax[-1][-1].get_legend_handles_labels()
    fig.legend(handles, ['$w_j$', r'$w_j^{eff}$', r'$w_j^{inv-eff}$', r'$\epsilon_j$'], ncol=1, loc=(0.92, 0.42))
    # fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'Metrics_evaluation_order{0}_{1}.pdf'.format(order, corr_metric)), dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contation process on temporal higher-order networks')
    parser.add_argument('--dataset', type=str, default='infectious', help='dataset')
    parser.add_argument('--beta', type=float, default=0.25, help='infectivity for pairwise interaction')
    parser.add_argument('--theta', type=float, default=2,
                        help='threshold for contagion to occur in hyperlink interaction')
    parser.add_argument('--R', type=int, default=100, help='the number of realizations in total')
    parser.add_argument('--n_arrays', type=int, default=10, help='the number of job arrays in slurm')
    parser.add_argument('--array_id', type=int, default=1, help='ID of job arrays in slurm')
    args = parser.parse_args()
    # h_tnet = hyperTN(args.dataset)
    # groupsize_statistics(h_tnet)
    # plot_prevalence(h_tnet)
    # average_shuffled_outputs(args.dataset, args.beta, num_r=100, num_s=10, theta=args.theta)
    # integrate_backbones(h_tnet, beta=args.beta, theta=args.theta, num_r=100)
    # backbone_vs_substrate(h_tnet)
    # all_datasets_backbone_vs_substrate(['infectious', 'ht09', 'highschool2013', 'primaryschool'], 1)
    # plot_backbone_comparison(h_tnet, order=4, theta=args.theta)  # the recall or weight distance as a function of beta
    # compare_diff_threshold_beta_1(h_tnet)
    # scatter_weights4subnets(h_tnet, theta=args.theta, ranking=False)
    # distance_weights4subnets(theta=args.theta)
    # scatter_weights_diff_thresholds(h_tnet, theta=args.theta, order=3, normalization=False, ranking=False)
    # compare_backbone_diff_t_heatmap(h_tnet, top_n=1000, theta=args.theta)
    # scatter_weights_topo_props(h_tnet, 0, metric='inverse weights', order=3)
    # for alpha in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     scatter_weights_topo_props_order2(h_tnet, alpha=alpha, phi=0, its=0)
    # scatter_weights_topo_props_order2(h_tnet, theta=args.theta, phi=0.0)
    # scatter_weights_topo_props_diff_order(h_tnet, theta=args.theta)
    # number_of_links_activated(h_tnet, weighted=True)
    # ratio_of_links_activated(h_tnet)
    # check_low_weight(h_tnet, args.beta)
    # check_2d(h_tnet)
    compare_diff_metrics()
