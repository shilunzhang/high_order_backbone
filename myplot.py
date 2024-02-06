import os.path
import pickle
from os import path
import json
import argparse
from functools import reduce
from collections import Counter
import numpy as np
from scipy.stats import rankdata, pearsonr, kendalltau, spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import fonts
from utils import *
from TNet import TN, PATH_TO_RESULTS
from hyperTNet import hyperTN
from centrality import *

# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams.update(fonts.neurips2021())
MARKERS = ['o', 'v', '^', '>', '<', 's', 'p', 'P', '*', 'h', 'X', 'D']
COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'cyan']
LINESTYLES = ['-', '--', '-.', ':']


def line_shaded(x, y, y_diff, ax, line_params=None):
    if line_params is None:
        line_params = dict()
    line = ax.plot(x, y, '-', linewidth=2, clip_on=False, **line_params)
    ax.fill_between(x, y + y_diff, y - y_diff, color='None', facecolor=line[0].get_color(), alpha=0.1)

    return line


def groupsize_statistics():
    datasets = ['infectious', 'primaryschool', 'highschool2012', 'highschool2013', 'ht09', 'SFHH', 'workplace15', 'hospital']

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.2))
    for d, dataset in enumerate(datasets):
        h_tnet = hyperTN(dataset)
        subnet, tmax = h_tnet.time_division(which='largest')
        groupsizes = list(reduce(lambda a, b: a + b, [[len(g) for g in h] for h in h_tnet.hypercontacts[:tmax]]))
        gs, freq = np.unique(groupsizes, return_counts=True)
        ax.plot(gs, freq/len(groupsizes), '-', marker=MARKERS[d], label=dataset+' '+str(freq[0]/len(groupsizes)))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of nodes in an event')
    ax.set_ylabel('Prob.')
    ax.set_xlim([1.9, 10])
    ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_xticklabels(['2', '3', '4', '5', '6', '7', '8', '9', '10'])
    fig.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'groupsize_statistics.pdf'), dpi=150)



def plot_prevalence(h_tnet: hyperTN):
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model')
    fig, ax = plt.subplots(1, len(beta), figsize=(2.2 * len(beta), 2.2))
    for i, bt in enumerate(beta):
        prevalence1 = np.load(path.join(res_path, 'beta1_{0:.2f}-beta2_{0:.2f}-theta_{1:.1f}'.format(bt, 1), 'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
        prevalence2 = np.load(path.join(res_path, 'beta1_{0:.2f}-beta2_{0:.2f}-theta_{1:.1f}'.format(bt, -1), 'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
        ax[i].plot(range(len(prevalence1)), prevalence1/h_tnet.n, c='black', clip_on=False, alpha=0.6, label=r'$\theta=1$')
        ax[i].plot(range(len(prevalence2)), prevalence2/h_tnet.n, c='maroon', clip_on=False, alpha=0.6, label=r'$\theta=h-1$')
        ax[i].plot([len(prevalence1)-1], [prevalence1[-1]/h_tnet.n], 'o', c='k', markersize=6, markeredgewidth=0, alpha=0.6, clip_on=False)
        ax[i].plot([len(prevalence2)-1], [prevalence2[-1]/h_tnet.n], 'o', c='maroon', markersize=6, markeredgewidth=0, alpha=0.6, clip_on=False)
        # for theta in ['1/2', '1/3', '1/4', '1/5', '1/6']:
        #     prevalence = np.load(path.join(res_path, 'beta1_{0:.2f}-beta2_{0:.2f}-theta_{1}'.format(bt, reduce(lambda x,y: x+'o'+y, theta.split('/'))),
        #                                     'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
        #     line = ax[i].plot(range(len(prevalence)), prevalence / h_tnet.n, '-', clip_on=False, alpha=0.6, label=r'$\theta={0}\cdot h$'.format(theta))
        #     ax[i].plot([len(prevalence) - 1], [prevalence[-1] / h_tnet.n], 'o', c=line[-1].get_color(), markersize=6, markeredgewidth=0, alpha=0.6, clip_on=False)
        ax[i].xaxis.set_tick_params(which='both', bottom=False)
        ax[i].yaxis.set_ticks([0.1 * t for t in range(len(ts)+1)])
        ax[i].yaxis.set_tick_params(which='both', direction='in')
        # ax[i].set_yscale('log')
        ax[i].set_ylim([0, 0.1 * len(ts)])
    ax[0].set_xlabel('time')
    ax[0].set_ylabel(r'prevalence')
    ax[0].legend(frameon=False, loc='upper left', borderpad=0.08, labelspacing=0.2)
    fig.tight_layout()
    fig.savefig(path.join(res_path, 'prevalence_evolution_all_theta.pdf'), dpi=200)
    # plt.show()

def prevalence_diff(xaxis='beta', yaxis='relative'):
    datasets = ['infectious', 'primaryschool', 'highschool2013', 'ht09', 'SFHH', 'workplace13', 'workplace15', 'malawi']
    beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.3))
    for d, dataset in enumerate(datasets):
        h_tnet = hyperTN(dataset)
        ts = h_tnet.time_division(which='all')
        res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model')
        prevalence_diff = []
        prevalence_theta_1 = []
        prevalence_theta_h_1 = []
        for b, bt in enumerate(beta):
            prevalence1 = np.load(path.join(res_path, 'beta1_{0:.3f}-beta2_{0:.3f}-theta_{1:.1f}'.format(bt, 1.0), 'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
            prevalence2 = np.load(path.join(res_path, 'beta1_{0:.3f}-beta2_{0:.3f}-theta_{1:.1f}'.format(bt, -1.0), 'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
            prevalence_diff.append((prevalence1[-1]-prevalence2[-1])/prevalence1[-1])
            prevalence_theta_1.append(prevalence1[-1]/h_tnet.n)
            prevalence_theta_h_1.append(prevalence2[-1]/h_tnet.n)
        ydata = prevalence_diff if yaxis == 'relative' else [(prevalence_theta_1[i]-prevalence_theta_h_1[i])/h_tnet.n for i in range(len(beta))]
        if xaxis == 'beta':
            ax.plot(beta, ydata, c=COLORS[d], clip_on=False, alpha=0.8, label=dataset)
            ax.set_xlabel(r'$\beta$')
        elif xaxis == 'prevalence_theta_1':
            ax.plot(prevalence_theta_1, ydata, c=COLORS[d], clip_on=False, alpha=0.8, label=dataset)
            ax.set_xlabel(r'$\rho(\Theta=1)/N$')
        elif xaxis == 'prevalence_theta_h_1':
            ax.plot(prevalence_theta_h_1, ydata, c=COLORS[d], clip_on=False, alpha=0.8, label=dataset)
            ax.set_xlabel(r'$\rho(\Theta=h-1)/N$')

    ax.set_ylabel(r'$\left(\rho(\Theta=1)-\rho(\Theta=h-1)\right )/\rho(\Theta=1)$' if yaxis=='relative' else r'$\left(\rho(\Theta=1)-\rho(\Theta=h-1)\right )/N$')
    ax.legend(frameon=False, borderpad=0.08, labelspacing=0.2)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'Prevalence_difference_{0}_{1}.pdf'.format(xaxis, yaxis)), dpi=200)

def integrate_backbones(h_tnet: hyperTN, beta, theta, num_r=100):
    if h_tnet.datatype == 'phy-contact':
        num_r = 5000
    ts = h_tnet.time_division(which='all')
    subnet = len(ts)

    theta = reduce(lambda x,y: x+'o'+y, theta.split('/'))
    for i, t in enumerate(ts):
        if i + 1 != subnet:
            continue
        for beta1 in [beta]:
            print(i, beta1)
            for beta2 in [beta]:
                res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                                     'beta1_{0:.3f}-beta2_{1:.3f}-theta_{2}'.format(beta1, beta2, theta))
                # if path.exists(path.join(res_path, 'T_0.{0}-backbone.pkl'.format(i + 1))):
                #     print('beta1_{0:.2f}-beta2_{1:.2f}-theta_{2}'.format(beta1, beta2, theta), '/T_0.{0}-backbone.pkl'.format(i + 1), ' existed..')
                #     if path.exists(path.join(res_path, 'T_0.{0}-prevalence_averaged.npy'.format(i + 1))):
                #         continue
                backbone = Counter()
                prevalence = np.zeros((t, num_r), dtype=float)
                for r in range(num_r+1, num_r + 5001):
                    if r/100 == r//100:
                        print(r, '/', num_r)
                    with open(path.join(res_path, 'T_0.{0}-backbone-r{1}.pkl'.format(i + 1, r)), 'rb') as f:
                        bb = pickle.load(f)
                        backbone.update(bb)
                    if os.path.exists(path.join(res_path, 'T_0.{0}-prevalence1d-r{1}.npy'.format(i + 1, r))):
                        prevalence[:, r - 5001] = np.load(path.join(res_path, 'T_0.{0}-prevalence1d-r{1}.npy'.format(i + 1, r))).astype(np.float64)
                    else:
                        prevalence[:, r - 5001] = np.loadtxt(
                            path.join(res_path, 'T_0.{0}-prevalence2d-r{1}.txt'.format(i + 1, r)), dtype=float).mean(
                            axis=1)
                # return backbone: normalize the weights.
                prevalence_average = prevalence.mean(axis=1)
                with open(path.join(res_path, 'T_0.{0}-backbone.pkl'.format(i + 1)), 'wb') as f:
                    pickle.dump(backbone, f)
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


def scatter_backbones_weights(h_tnet: hyperTN, beta, theta, ax, order, normalization=False, ranking=False):
    r_beta = []
    subnet = len(h_tnet.time_division(which='all'))
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                        'beta1_{0:.3f}-beta2_{1:.3f}-theta_{2}'.format(beta, beta, 1.0),
                        'T_0.{0}-backbone.pkl'.format(subnet)), 'rb') as f:
        backbone1 = pickle.load(f)
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                        'beta1_{0:.3f}-beta2_{1:.3f}-theta_{2}'.format(beta, beta, theta),
                        'T_0.{0}-backbone.pkl'.format(subnet)), 'rb') as f:
        backbone2 = pickle.load(f)
    n_realizations = 5000
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
        ax.set_xscale('symlog', linthresh=1/5000)
        ax.set_yscale('symlog', linthresh=1/5000)
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([xmin, xmax])


def scatter_weights_diff_thresholds(h_tnet: hyperTN, theta, order, normalization, ranking=False):
    beta = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, ax = plt.subplots(2, len(beta), figsize=(1.7 * len(beta), 1.7*2))
    for j, b in enumerate(beta):
        # ax[i][j].axis('equal')
        scatter_backbones_weights(h_tnet, beta=b, theta=theta, ax=ax[0][j], order=2, normalization=normalization, ranking=ranking)
        scatter_backbones_weights(h_tnet, beta=b, theta=theta, ax=ax[1][j], order=3, normalization=normalization, ranking=ranking)
    fig.tight_layout()
    suffix = ''
    suffix += '_normalization' if normalization else ''
    suffix += '_rank' if ranking else ''
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model',
                          'Weights_scatter_diff_thresholds_theta_{0}{1}_.pdf'.format(theta, suffix)), dpi=100)


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

def distribution_log_ratio_weights(h_tnet: hyperTN, order, subnet=-1):  # TODO
    eps = 10**(-8)
    beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    beta = [e for e in beta]

    ts = h_tnet.time_division(which='all')
    agg_net = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for b, bt in enumerate(beta):
        backbone1 = h_tnet.return_backbone(params={'beta': bt, 'theta': 1.0}, subnet=h_tnet.time_division(which='largest')[0])
        backbone2 = h_tnet.return_backbone(params={'beta': bt, 'theta': -1.0}, subnet=h_tnet.time_division(which='largest')[0])
        log_ratios = [np.log10((backbone1[hl]+eps if hl in backbone1 else eps)/(backbone2[hl]+eps if hl in backbone2 else eps)) for hl in agg_net if len(hl)==order]
        ax.hist(log_ratios, bins=20, color=COLORS[b], histtype='step', weights=np.ones(len(log_ratios))/len(log_ratios), label=r'$\beta={0}$'.format(bt))

    ax.set_xlabel('log ratio')
    ax.set_ylabel('Prob.')
    fig.legend(frameon=False)
    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'log_ratio_weights_distribution_order{0}.pdf'.format(order)), dpi=200)

def corr_coef_backbone_metric(h_tnet: hyperTN, params: dict, subnet: int, order: int, metric: str, corr_metric: str, coef=1.0, alpha=0):
    global metric_dict
    if metric == 'linkweight':
        metric_dict = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:h_tnet.time_division(which='all')[subnet]])
    # elif metric == 'local_subplink_prod_metric':
    #     metric_dict = local_subplink_prod_metric(h_tnet, subnet)
    # elif metric == 'local_adjplink_prod_metric':
    #     metric_dict = local_adjplink_prod_metric(h_tnet, subnet)
    # elif metric == 'local_adjlink_division_metric':
    #     metric_dict = local_adjlink_division_metric(h_tnet, subnet)
    # elif metric == 'local_subsuplink_division_metric':
    #     metric_dict = local_subsuplink_division_metric(h_tnet, subnet)
    # elif metric == 'local_subplink_prod_temporal_metric':
    #     metric_dict = local_subplink_prod_temporal_metric(h_tnet, subnet)
    # elif metric == 'local_adjplink_prod_temporal_metric':
    #     metric_dict = local_adjplink_prod_temporal_metric(h_tnet, subnet)
    # elif metric == 'local_adjlink_division_temporal_metric':
    #     metric_dict = local_adjlink_division_temporal_metric(h_tnet, subnet)
    # elif metric == 'local_subsuplink_division_temporal_metric':
    #     metric_dict = local_subsuplink_division_temporal_metric(h_tnet, subnet)
    else:
        with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'centrality', 'T_0.{0}-{1}.pkl'.format(subnet+1, metric)), 'rb') as f:
            metric_dict = pickle.load(f)

    metric_dict = {k:v for k,v in metric_dict.items() if len(k) == order}
    backbone = h_tnet.return_backbone(params, subnet=subnet)
    data = [(backbone[k], metric_dict[k]) if k in backbone else (0, metric_dict[k]) for k in metric_dict]

    if corr_metric == 'kendalltau':
        return kendalltau([e[0] for e in data], [e[1] for e in data])[0]
    elif corr_metric == 'pearsonr':
        return pearsonr([e[0] for e in data], [e[1] for e in data])[0]

def compare_diff_metrics_corr(theta, order=3, corr_metric='kendalltau', xaxis='beta'):
    dataset_phy_contacts = ['infectious', 'primaryschool', 'highschool2012', 'highschool2013']
    # dataset_sci_collab = ['q-bio', 'q-fin', 'hep-lat', 'nucl-th']
    dataset_sci_collab = ['ht09', 'SFHH', 'workplace15', 'hospital']
    # metrics = ['linkweight', 'local_subplink_prod_metric', 'local_subplink_division_metric', 'local_adjplink_prod_metric', 'local_adjplink_division_metric', 'local_adjlink_division_metric', 'local_subsuplink_division_metric']
    # metrics = metrics + ['local_subplink_prod_temporal_metric', 'local_subplink_division_temporal_metric', 'local_adjplink_prod_temporal_metric', 'local_adjplink_division_temporal_metric', 'local_adjlink_division_temporal_metric', 'local_subsuplink_division_temporal_metric']
    metrics = ['linkweight', 'time_independent_link_local_metric_subplink_alpha1.0', 'time_independent_link_local_metric_subplink_alpha-1.0',
               'time_independent_link_local_metric_adjplink_alpha1.0', 'time_independent_link_local_metric_adjplink_alpha-1.0']
    metrics = metrics + ['time_dependent_link_local_metric_subplink_alpha1.0', 'time_dependent_link_local_metric_subplink_alpha-1.0', 'time_dependent_link_local_metric_adjplink_alpha1.0', 'time_dependent_link_local_metric_adjplink_alpha-1.0']
    # metrics = ['local_subplink_division_metric']
    beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, ax = plt.subplots(2, 4, figsize=(len(dataset_phy_contacts) * 3.2*0.55, 2.6 * 2.6 *0.55), sharey=True, sharex=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.33, top=0.97)
    for idx in range(ax.shape[1]):
        h_tnet1 = hyperTN(dataset_phy_contacts[idx])  # plot face-to-face datasets
        ts1 = h_tnet1.time_division(which='all')
        corr_coef1, prev1 = [], []
        for bt in beta:
            backbone1 = h_tnet1.return_backbone(params={'beta': bt, 'theta': theta}, subnet=len(ts1)-1)
            prev1.append(np.sum(list(backbone1.values()))/h_tnet1.n**2)
            corr_coef1.append([corr_coef_backbone_metric(h_tnet1, {'beta': bt, 'theta': theta}, subnet=len(ts1)-1, order=order, metric=metric, corr_metric=corr_metric, coef=bt) for metric in metrics])
        for m in range(len(metrics)):
            if 'time_dependent' in metrics[m]:
                ax[0][idx].plot(beta if xaxis=='beta' else prev1, [e[m] for e in corr_coef1], '-', linewidth=1, c=COLORS[m-4], label=metrics[m], alpha=0.8)
            else:
                ax[0][idx].plot(beta if xaxis=='beta' else prev1, [e[m] for e in corr_coef1], ':', linewidth=1.4, c=COLORS[m], label=metrics[m], alpha=0.8)
        h_tnet2 = hyperTN(dataset_sci_collab[idx])  # plot arxiv datasets
        ts2 = h_tnet2.time_division(which='all')
        corr_coef2, prev2 = [], []
        for bt in beta:
            backbone2 = h_tnet2.return_backbone(params={'beta': bt, 'theta': theta}, subnet=len(ts2)-1)
            prev2.append(np.sum(list(backbone2.values()))/h_tnet2.n**2)
            corr_coef2.append([corr_coef_backbone_metric(h_tnet2, {'beta': bt, 'theta': theta}, subnet=len(ts2) - 1,
                                order=order, metric=metric, corr_metric=corr_metric, coef=bt) for metric in metrics])
        for m in range(len(metrics)):
            if 'time_dependent' in metrics[m]:
                ax[1][idx].plot(beta if xaxis=='beta' else prev2, [e[m] for e in corr_coef2], '-', linewidth=1, c=COLORS[m-4], label=metrics[m], alpha=0.8)
            else:
                ax[1][idx].plot(beta if xaxis=='beta' else prev2, [e[m] for e in corr_coef2], ':', linewidth=1.4, c=COLORS[m], label=metrics[m], alpha=0.8)
        ax[0][idx].plot([0.0008, 1.1], [0, 0], '--', linewidth=0.6, c='grey')
        ax[1][idx].plot([0.0008, 1.1], [0, 0], '--', linewidth=0.6, c='grey')
        ax[1][idx].set_xlabel(r'$\beta$' if xaxis=='beta' else r'$\rho$', fontsize=9)
        # ax[0][idx].set_xticks([0.001, 0.01, 0.1, 1.0], ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'], fontsize=6)
        ax[0][idx].xaxis.set_tick_params(labelsize=6)
        ax[1][idx].xaxis.set_tick_params(labelsize=6)
        # ax[1][idx].set_xticks([0.001, 0.01, 0.1, 1.0], ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'], fontsize=6)
        # ax[0][idx].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # ax[0][idx].set_xlim([-0.05, 1.05])
        ax[1][idx].set_xscale('log')
        plt.text(.97, .03, dataset_phy_contacts[idx], ha='right', va='bottom', transform=ax[0][idx].transAxes, fontsize=10, fontweight='bold')
        plt.text(.97, .03, dataset_sci_collab[idx], ha='right', va='bottom', transform=ax[1][idx].transAxes, fontsize=10, fontweight='bold')
    ax[0][0].set_ylim([-1.0, 1.0])
    ax[1][0].set_ylim([-1.0, 1.0])
    ax[0][0].set_ylabel(r'Kendall correlation', fontsize=9)
    ax[1][0].set_ylabel(r'Kendall correlation', fontsize=9)
    ax[0][0].set_yticks([-1, -0.5, 0, 0.5, 1.0], ['$-1.0$', '$-0.5$', '$0.0$', '$0.5$', '$1.0$'], fontsize=6)
    ax[1][0].set_yticks([-1, -0.5, 0, 0.5, 1.0], ['$-1.0$', '$-0.5$', '$0.0$', '$0.5$', '$1.0$'], fontsize=6)
    handles, labels = ax[-1][-1].get_legend_handles_labels()
    # fig.legend(handles, ['$w_j$', r'$w_j^{eff}$', r'$w_j^{inv-eff}$', r'$\epsilon_j$', r'$\tilde{w}_j$', r'$w_j^{exp-decay-eff}$'], ncol=1, loc=(0.89, 0.42), fontsize='large')
    labels = [r'$w_j$', r'$\xi_j^{sub-pairwise}(\alpha=1)$', r'$\xi_j^{sub-pairwise}(\alpha=-1)$', r'$\xi_j^{adj-pairwise}(\alpha=1)$',
              r'$\xi_j^{adj-pairwise}(\alpha=-1)$',
              r'$\Xi_j^{sub-pairwise}(\alpha=1)$', r'$\Xi_j^{sub-pairwise}(\alpha=-1)$', r'$\Xi_j^{adj-pairwise}(\alpha=1)$',
              r'$\Xi_j^{adj-pairwise}(\alpha=-1)$']
    fig.legend(handles, labels, fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, columnspacing=0.5)
    # fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'Metrics_evaluation_order{0}_theta{1}_{2}_face2face_all_{3}.pdf'.format(order, '{0:.1f}'.format(theta) if isinstance(theta, float) else reduce(lambda x,y: x+'o'+y, theta.split('/')), corr_metric, xaxis)), dpi=200)

def compare_diff_exponent_coef_metrics_corr(theta, order=3, corr_metric='kendalltau'):
    dataset_phy_contacts = ['infectious', 'primaryschool', 'highschool2013', 'ht09']
    # dataset_sci_collab = ['q-bio', 'q-fin', 'hep-lat', 'nucl-th']
    dataset_sci_collab = ['SFHH', 'workplace13', 'workplace15', 'malawi']
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    coefs = [0.01, 0.1, 0.2, 1.0]

    fig, ax = plt.subplots(2, 4, figsize=(len(dataset_phy_contacts) * 3.2, 2 * 2.5), sharey=True, sharex=True)
    plt.subplots_adjust(left=0.06, right=0.87, bottom=0.11, top=0.97)
    for idx in range(ax.shape[1]):
        h_tnet1 = hyperTN(dataset_phy_contacts[idx])  # plot face-to-face datasets
        ts1 = h_tnet1.time_division(which='all')
        corr_coef1 = []
        for bt in beta:
            corr_coef1.append([corr_coef_backbone_metric(h_tnet1, {'beta': bt, 'theta': theta}, subnet=len(ts1)-1,
                                order=order, metric='local_effective_sum_exponent_normalized_metric', corr_metric=corr_metric, coef=coef) for coef in coefs]
                              + [corr_coef_backbone_metric(h_tnet1, {'beta': bt, 'theta': theta}, subnet=len(ts1) - 1,
                                                           order=order,
                                                           metric='local_effective_sum_exponent_normalized_metric',
                                                           corr_metric=corr_metric, coef=bt)])
        for m in range(len(coefs)+1):
            ax[0][idx].plot(beta, [e[m] for e in corr_coef1], '-o')
        h_tnet2 = hyperTN(dataset_sci_collab[idx])  # plot arxiv datasets
        ts2 = h_tnet2.time_division(which='all')
        corr_coef2 = []
        for bt in beta:
            corr_coef2.append([corr_coef_backbone_metric(h_tnet2, {'beta': bt, 'theta': theta}, subnet=len(ts2) - 1,
                                order=order, metric='local_effective_sum_exponent_normalized_metric', corr_metric=corr_metric, coef=coef) for coef in coefs]
                              +[corr_coef_backbone_metric(h_tnet2, {'beta': bt, 'theta': theta}, subnet=len(ts2) - 1,
                                order=order, metric='local_effective_sum_exponent_normalized_metric', corr_metric=corr_metric, coef=bt)])
        for m in range(len(coefs)+1):
            ax[1][idx].plot(beta, [e[m] for e in corr_coef2], '-o', label=(coefs+[r'$\beta$'])[m])
        ax[0][idx].plot([-0.05, 1.05], [0, 0], '--', c='grey')
        ax[1][idx].plot([-0.05, 1.05], [0, 0], '--', c='grey')
        ax[1][idx].set_xlabel(r'$\beta$')
        ax[0][idx].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[0][idx].set_xlim([-0.05, 1.05])
        plt.text(.97, .97, dataset_phy_contacts[idx], ha='right', va='top', transform=ax[0][idx].transAxes, fontsize='large', fontweight='bold')
        plt.text(.97, .97, dataset_sci_collab[idx], ha='right', va='top', transform=ax[1][idx].transAxes, fontsize='large', fontweight='bold')
    ax[0][0].set_ylim([-1.0, 1.0])
    ax[1][0].set_ylim([-1.0, 1.0])
    ax[0][0].set_ylabel('Kendall tau')
    ax[1][0].set_ylabel('Kendall tau')
    handles, labels = ax[-1][-1].get_legend_handles_labels()
    fig.legend(handles, coefs+[r'$\beta$'], ncol=1, loc=(0.89, 0.42), fontsize='small')
    # fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'Metrics_evaluation_order{0}_theta{1}_{2}_face2face_0hop_exponent_coefs.pdf'.format(order, '{0:.1f}'.format(theta) if isinstance(theta, float) else reduce(lambda x,y: x+'o'+y, theta.split('/')), corr_metric)), dpi=200)


def compare_time_decayed_metrics_corr(theta, order=3, corr_metric='kendalltau'):
    dataset_phy_contacts = ['infectious', 'primaryschool', 'highschool2013', 'ht09']
    dataset_sci_collab = ['q-bio', 'q-fin', 'hep-lat', 'nucl-th']
    metric_alpha = [0, 0.5, 1.0, 1.5, 2.0]
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, ax = plt.subplots(2, 4, figsize=(len(dataset_phy_contacts) * 3.2, 2 * 2.5), sharey=True, sharex=True)
    plt.subplots_adjust(left=0.06, right=0.87, bottom=0.11, top=0.97)
    for idx in range(ax.shape[1]):
        h_tnet1 = hyperTN(dataset_phy_contacts[idx])  # plot face-to-face datasets
        ts1 = h_tnet1.time_division(which='all')
        corr_coef1 = []
        for bt in beta:
            corr_coef1.append([corr_coef_backbone_metric(h_tnet1, {'beta': bt, 'theta': theta}, subnet=len(ts1)-1, order=order, metric='linkweight_timedecayed', corr_metric=corr_metric, alpha=alp) for alp in metric_alpha])
        for m in range(len(metric_alpha)):
            ax[0][idx].plot(beta, [e[m] for e in corr_coef1], '-', marker=MARKERS[m], c=COLORS[m], label=r'$\alpha={0}$'.format(metric_alpha[m]))
        h_tnet2 = hyperTN(dataset_sci_collab[idx])  # plot arxiv datasets
        ts2 = h_tnet2.time_division(which='all')
        corr_coef2 = []
        for bt in beta:
            corr_coef2.append([corr_coef_backbone_metric(h_tnet2, {'beta': bt, 'theta': theta}, subnet=len(ts2) - 1,
                                order=order, metric='linkweight_timedecayed', corr_metric=corr_metric, alpha=alp) for alp in metric_alpha])
        for m in range(len(metric_alpha)):
            ax[1][idx].plot(beta, [e[m] for e in corr_coef2], '-', marker=MARKERS[m], c=COLORS[m], label=r'$\alpha={0}$'.format(metric_alpha[m]))
        ax[0][idx].plot([-0.05, 1.05], [0, 0], '--', c='grey')
        ax[1][idx].plot([-0.05, 1.05], [0, 0], '--', c='grey')
        ax[1][idx].set_xlabel(r'$\beta$')
        ax[0][idx].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[0][idx].set_xlim([-0.05, 1.05])
        plt.text(.97, .97, dataset_phy_contacts[idx], ha='right', va='top', transform=ax[0][idx].transAxes, fontsize='large', fontweight='bold')
        plt.text(.97, .97, dataset_sci_collab[idx], ha='right', va='top', transform=ax[1][idx].transAxes, fontsize='large', fontweight='bold')
    ax[0][0].set_ylim([-1.0, 1.0])
    ax[1][0].set_ylim([-1.0, 1.0])
    ax[0][0].set_ylabel('Kendall tau')
    ax[1][0].set_ylabel('Kendall tau')
    handles, labels = ax[-1][-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=1, loc=(0.89, 0.42), fontsize='large')
    # fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'Timedecayed_Metrics_evaluation_order{0}_theta{1}_{2}.pdf'.format(order, '{0:.1f}'.format(theta) if isinstance(theta, float) else reduce(lambda x,y: x+'o'+y, theta.split('/')), corr_metric)), dpi=200)

def colormap2d(h_tnet: hyperTN, subnet):
    ts = h_tnet.time_division(which='all')
    beta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    link_weight = h_tnet.aggregate_hyperTN(hcontacts=h_tnet.hypercontacts[:ts[subnet]])
    eff_link_weight = local_effective_sum_metric(h_tnet, subnet=subnet)
    square_eff_link_weight = local_effective_square_sum_metric(h_tnet, subnet=subnet)

    fig, ax = plt.subplots(2, len(beta), figsize=(len(beta) * 3.2, 2 * 2.5))
    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.11, top=0.97)
    for b, bt in enumerate(beta):
        backbone1 = h_tnet.return_backbone({'beta': bt, 'theta': 1.0}, subnet=subnet)
        backbone2 = h_tnet.return_backbone({'beta': bt, 'theta': -1.0}, subnet=subnet)
        data1 = np.array([(link_weight[hl], eff_link_weight[hl] if hl in eff_link_weight else 0, backbone1[hl] if hl in backbone1 else 0) for hl in link_weight if len(hl) == 3], dtype=np.float64)
        data2 = np.array([(eff_link_weight[hl] if hl in eff_link_weight else 0, square_eff_link_weight[hl] if hl in square_eff_link_weight else 0, backbone2[hl] if hl in backbone2 else 0) for hl in link_weight if len(hl) == 3], dtype=np.float64)

        occ_cell, xbins, ybins = np.histogram2d(data1[:, 0], data1[:, 1], bins=[np.logspace(np.log10(data1[:, 0].min()), np.log10(data1[:, 0].max()), 6), np.logspace(np.log10(data1[data1[:, 1]>0, 1].min()), np.log10(data1[:, 1].max()), 12)], density=False)
        sum_cell, xbins, ybins = np.histogram2d(data1[:, 0], data1[:, 1], bins=[np.logspace(np.log10(data1[:, 0].min()), np.log10(data1[:, 0].max()), 6), np.logspace(np.log10(data1[data1[:, 1]>0, 1].min()), np.log10(data1[:, 1].max()), 12)], density=False, weights=data1[:, 2])
        mean_cell = -np.ones(occ_cell.shape, dtype=np.float64)
        mean_cell[occ_cell != 0] = sum_cell[occ_cell != 0] / occ_cell[occ_cell != 0]
        X, Y = np.meshgrid(xbins, ybins)
        print(xbins.shape, ybins.shape)
        print(X.shape)
        print(Y.shape)
        ax[0][b].pcolormesh(X, Y, np.ma.masked_where(mean_cell<0, mean_cell).T, cmap='Blues', norm='log')
        ax[0][b].set_xscale('log')
        ax[0][b].set_yscale('log')

        occ_cell, xbins, ybins = np.histogram2d(data2[:, 0], data2[:, 1], bins=[np.logspace(np.log10(data2[data2[:, 0]>0, 1].min()), np.log10(data2[:, 0].max()), 6), np.logspace(np.log10(data2[data2[:, 1]>0, 1].min()), np.log10(data2[:, 1].max()), 12)], density=False)
        sum_cell, xbins, ybins = np.histogram2d(data2[:, 0], data2[:, 1], bins=[np.logspace(np.log10(data2[data2[:, 0]>0, 1].min()), np.log10(data2[:, 0].max()), 6), np.logspace(np.log10(data2[data2[:, 1]>0, 1].min()), np.log10(data2[:, 1].max()), 12)], density=False, weights=data2[:, 2])
        mean_cell = -np.ones(occ_cell.shape, dtype=np.float64)
        mean_cell[occ_cell != 0] = sum_cell[occ_cell != 0] / occ_cell[occ_cell != 0]
        X, Y = np.meshgrid(xbins, ybins)
        ax[1][b].pcolormesh(X, Y, np.ma.masked_where(mean_cell<0, mean_cell).T, cmap='Blues', norm='log')
        ax[1][b].set_xscale('log')
        ax[1][b].set_yscale('log')

    fig.savefig(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'colarmap2d.pdf'), dpi=200)

def prevalence_comparison():
    import matplotlib.gridspec as gridspec
    datasets = ['infectious', 'primaryschool', 'highschool2012', 'highschool2013', 'ht09', 'SFHH', 'workplace15', 'hospital']
    beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
            0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig = plt.figure(figsize=(5, 3.8))
    gs = gridspec.GridSpec(2, 6, height_ratios=[3, 2])

    ax0 = fig.add_subplot(gs[0, 1:5])
    ax1 = fig.add_subplot(gs[1, :3])
    ax2 = fig.add_subplot(gs[1, 3:])
    plt.subplots_adjust(left=0.11, bottom=0.11, right=0.7, top=0.98, wspace=0.4, hspace=0.3)
    for d, ds in enumerate(datasets):
        h_tnet = hyperTN(ds)
        ts = h_tnet.time_division(which='all')
        res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model')
        prevalence = np.load(path.join(res_path, 'beta1_{0:.3f}-beta2_{0:.3f}-theta_{1:.1f}'.format(1.0, 1), 'prevalence1d.npy'), allow_pickle=True)
        ax0.plot(np.arange(len(prevalence))/len(prevalence), prevalence/h_tnet.n, '-', c=COLORS[d], label=ds)
        prevalence_diff, prevalence_theta_1, prevalence_theta_h_1 = [], [], []
        for b, bt in enumerate(beta):
            prevalence1 = np.load(path.join(res_path, 'beta1_{0:.3f}-beta2_{0:.3f}-theta_{1:.1f}'.format(bt, 1.0),
                                            'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
            prevalence2 = np.load(path.join(res_path, 'beta1_{0:.3f}-beta2_{0:.3f}-theta_{1:.1f}'.format(bt, -1.0),
                                            'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
            prevalence_diff.append((prevalence1[-1] - prevalence2[-1]) / prevalence1[-1])
            prevalence_theta_1.append(prevalence1[-1] / h_tnet.n)
            prevalence_theta_h_1.append(prevalence2[-1] / h_tnet.n)
        ydata = prevalence_diff
        ax1.plot(beta, ydata, c=COLORS[d], clip_on=False)
        ax2.plot(prevalence_theta_1, ydata, c=COLORS[d], clip_on=False)

    ax0.set_xlabel(r'$t/T$', fontsize=8, labelpad=0.0)
    ax1.set_xlabel(r'$\beta$', fontsize=8)
    ax2.set_xlabel(r'$\frac{\rho(\Theta=1)}{N}$', fontsize=8, labelpad=0.5)
    ax0.set_ylabel(r'$\rho(\Theta=1,\beta=1)$', fontsize=8)
    ax1.set_ylabel(r'$\frac{\rho(\Theta=1)-\rho(\Theta=d-1)}{\rho(\Theta=1)}$', fontsize=8)
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax0.set_xlim([-0.05, 1.05])
    ax0.set_ylim([-0.05, 1.05])
    ax1.set_ylim([0, 0.7])
    ax2.set_ylim([0, 0.7])
    ax1.set_xticks([0.001, 0.01, 0.1, 1.0])
    ax2.set_xticks([0.001, 0.01, 0.1, 1.0])
    ax0.xaxis.set_tick_params(labelsize=6)
    ax0.yaxis.set_tick_params(labelsize=6)
    ax1.xaxis.set_tick_params(labelsize=6)
    ax1.yaxis.set_tick_params(labelsize=6)
    ax2.xaxis.set_tick_params(labelsize=6)
    ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax2.set_yticklabels([])
    # ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
    # ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.5), ncol=1)
    # fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'Fig_prevalence.pdf'), dpi=200)

def compare_relative_weight(xvar='prevalence', normalize=True):
    dataset_phy_contacts = ['infectious', 'primaryschool', 'highschool2012', 'highschool2013']
    dataset_sci_collab = ['ht09', 'SFHH', 'workplace15', 'hospital']
    beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    theta = ['1.0', '-1.0']
    orders = [2, 3]

    fig, ax = plt.subplots(2, 4, figsize=(7, 3.1), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.06, right=0.8, bottom=0.11, top=0.90)
    for idx in range(ax.shape[1]):
        h_tnet1 = hyperTN(dataset_phy_contacts[idx])  # plot face-to-face datasets
        ts1 = h_tnet1.time_division(which='all')
        for idtt, tt in enumerate(theta):
            weights_sum, prev = [], []
            for bt in beta:
                backbone = h_tnet1.return_backbone({'beta': bt, 'theta': tt}, subnet=len(ts1)-1)
                weights_sum.append([sum([backbone[k] for k in backbone if len(k) == od])/h_tnet1.n for od in orders])
                prev.append(sum([backbone[k] for k in backbone])/h_tnet1.n)
            for iod, od in enumerate(orders):
                if xvar == 'beta':
                    ax[0][idx].plot(beta, [weights_sum[i][iod]/(prev[i] if normalize else 1) for i in range(len(beta))], linestyle=LINESTYLES[idtt], c=COLORS[iod], clip_on=False)
                elif xvar =='prevalence':
                    ax[0][idx].plot([e/h_tnet1.n for e in prev], [weights_sum[i][iod] / (prev[i] if normalize else 1) for i in range(len(beta))], linestyle=LINESTYLES[idtt], c=COLORS[iod], clip_on=False)
        ax[0][idx].set_xscale('log')
        ax[0][idx].set_title(dataset_phy_contacts[idx], fontsize=8, fontweight='bold', pad=0.5)
        ax[0][idx].tick_params(direction='in', length=2.2, width=0.7)

        h_tnet2 = hyperTN(dataset_sci_collab[idx])  # plot face-to-face datasets
        ts2 = h_tnet2.time_division(which='all')
        for idtt, tt in enumerate(theta):
            weights_sum, prev = [], []
            for bt in beta:
                backbone = h_tnet2.return_backbone({'beta': bt, 'theta': tt}, subnet=len(ts2) - 1)
                weights_sum.append([sum([backbone[k] for k in backbone if len(k) == od])/h_tnet2.n for od in orders])
                prev.append(sum([backbone[k] for k in backbone])/h_tnet2.n)
            for iod, od in enumerate(orders):
                if xvar == 'beta':
                    ax[1][idx].plot(beta, [weights_sum[i][iod] / (prev[i] if normalize else 1) for i in range(len(beta))], linestyle=LINESTYLES[idtt], c=COLORS[iod], clip_on=False, label=r'$\Theta={0}, d={1}$'.format('1' if tt=='1.0' else 'd-1', od))
                elif xvar =='prevalence':
                    ax[1][idx].plot([e / h_tnet2.n for e in prev], [weights_sum[i][iod] / (prev[i] if normalize else 1) for i in range(len(beta))], linestyle=LINESTYLES[idtt], c=COLORS[iod], clip_on=False, label=r'$\Theta={0}, d={1}$'.format('1' if tt=='1.0' else 'd-1', od))
        # if xvar == 'prevalence':
        #     # ax[0][idx].set_xscale('log')
        #     # ax[1][idx].set_xscale('log')
        #     pass
        # else:
        #     ax[1][idx].set_xlim([-0.05, 1.05])
        # ax[1][idx].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # ax[0][idx].set_yscale('symlog', linthresh=1/h_tnet1.n)
        # ax[1][idx].set_yscale('symlog', linthresh=1/h_tnet2.n)
        ax[1][idx].set_xscale('log')
        ax[1][idx].set_xticks([0.001, 0.01, 0.1, 1.0])
        ax[1][idx].set_xlabel(r'$\beta$' if xvar == 'beta' else 'prevalence', fontsize=8, labelpad=0.5)
        ax[1][idx].xaxis.set_tick_params(labelsize=6)
        ax[1][idx].tick_params(direction='in', length=2.2, width=0.7)
        ax[1][idx].set_title(dataset_sci_collab[idx], fontsize=8, fontweight='bold', pad=0.5)
        # plt.text(.97, .50, dataset_phy_contacts[idx], ha='right', va='top', transform=ax[0][idx].transAxes,
        #          fontsize=8, fontweight='bold')
        # plt.text(.97, .50, dataset_sci_collab[idx], ha='right', va='top', transform=ax[1][idx].transAxes,
        #          fontsize=8, fontweight='bold')
    ax[0][0].set_ylim([-0.05, 1.05])
    ax[1][0].set_ylim([-0.05, 1.05])
    ax[0][0].yaxis.set_tick_params(labelsize=6)
    ax[1][0].yaxis.set_tick_params(labelsize=6)
    ax[0][0].set_ylabel('Sum of weight', fontsize=8)
    ax[1][0].set_ylabel('Sum of weight', fontsize=8)
    handles, labels = ax[-1][-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=1, loc=(0.82, 0.42), fontsize=7, frameon=False)
    # fig.tight_layout()
    fig.savefig(
        path.join(PATH_TO_FIGS, 'Fig_order_induced_prevalence{0}.pdf'.format('_'+xvar)),
        dpi=200)

def corr_diff_theta(corr_measure_func):
    datasets = ['infectious', 'primaryschool', 'highschool2012', 'highschool2013', 'ht09', 'SFHH', 'workplace15', 'hospital']
    beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.2), sharex=True)
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.3, top=0.92, wspace=0.07)
    for d, dataset in enumerate(datasets):
        h_tnet = hyperTN(dataset)
        subset, tmax = h_tnet.time_division(which='largest')
        agg_net = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:tmax])
        print(dataset, len(agg_net), len([1 for hl in agg_net if len(hl)==2]), len([1 for hl in agg_net if len(hl)==3]))
        corrs, corrs1, corrs2 = [], [], []
        for bt in beta:
            backbone1 = h_tnet.return_backbone({'beta': bt, 'theta': 1.0}, subnet=subset)
            backbone2 = h_tnet.return_backbone({'beta': bt, 'theta': -1.0}, subnet=subset)
            data = [[backbone1[hl] if hl in backbone1 else 0, backbone2[hl] if hl in backbone2 else 0] for hl in agg_net]
            data1 = [[backbone1[hl] if hl in backbone1 else 0, backbone2[hl] if hl in backbone2 else 0] for hl in agg_net if len(hl) == 2]
            data2 = [[backbone1[hl] if hl in backbone1 else 0, backbone2[hl] if hl in backbone2 else 0] for hl in agg_net if len(hl) == 3]
            corrs.append(corr_measure_func([e[0] for e in data], [e[1] for e in data])[0])
            corrs1.append(corr_measure_func([e[0] for e in data1], [e[1] for e in data1])[0])
            corrs2.append(corr_measure_func([e[0] for e in data2], [e[1] for e in data2])[0])
            # corrs.append(overlap_top_N([e[0] for e in data], [e[1] for e in data], len(data)//2))
            # corrs1.append(overlap_top_N([e[0] for e in data1], [e[1] for e in data1], len(data1)//2))
            # corrs2.append(overlap_top_N([e[0] for e in data2], [e[1] for e in data2], len(data2)//2))
        ax[0].plot(beta, corrs, '-', label=dataset)
        ax[1].plot(beta, corrs1, '-')
        ax[2].plot(beta, corrs2, '-')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')
    ax[0].set_title('Order $d\geq 2$', fontsize=8, pad=0.8)
    ax[1].set_title('Order $d=2$', fontsize=8, pad=0.8)
    ax[2].set_title('Order $d\geq3$', fontsize=8, pad=0.8)
    ax[0].set_xticks([0.001, 0.01, 0.1, 1.0])
    ax[1].set_xticks([0.001, 0.01, 0.1, 1.0])
    ax[2].set_xticks([0.001, 0.01, 0.1, 1.0])
    ax[0].tick_params(direction='in', length=2.2, width=0.7)
    ax[1].tick_params(direction='in', length=2.2, width=0.7)
    ax[2].tick_params(direction='in', length=2.2, width=0.7)
    ax[0].xaxis.set_tick_params(labelsize=6)
    ax[1].xaxis.set_tick_params(labelsize=6)
    ax[2].xaxis.set_tick_params(labelsize=6)
    ax[0].yaxis.set_tick_params(labelsize=6)
    ax[1].yaxis.set_tick_params(labelsize=6)
    ax[2].yaxis.set_tick_params(labelsize=6)
    ax[0].set_xlabel(r'$\beta$', fontsize=8, labelpad=0.8)
    ax[1].set_xlabel(r'$\beta$', fontsize=8, labelpad=0.8)
    ax[2].set_xlabel(r'$\beta$', fontsize=8, labelpad=0.8)
    # ax[0].set_ylabel(r'Kendall correlation', fontsize=8)
    ax[0].set_ylabel(r'Spearman r', fontsize=8)
    # ax[0].set_ylabel(r'Overlap')
    ax[0].set_ylim([0, 1.05])
    fig.legend(frameon=False, ncols=4, loc='lower center', bbox_to_anchor=(0.5, 0.01), borderpad=0.05, columnspacing=0.5, fontsize=7)
    # fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'Fig_weight_corr_spearmanr.pdf'), dpi=200)
    # fig.savefig(path.join(PATH_TO_FIGS, 'Overlap_diff_theta{0}.pdf'.format(f'_order{order}' if order else '')), dpi=200)

def netsci_fig():
    datasets = ['infectious', 'primaryschool', 'highschool2012', 'highschool2013', 'ht09', 'SFHH', 'workplace15', 'hospital']
    beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
            0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, ax = plt.subplots(1, 3, figsize=(7, 2.5))
    # for d, ds in enumerate(datasets):
    #     h_tnet = hyperTN(ds)
    #     ts = h_tnet.time_division(which='all')
    #     res_path = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model')
    #     prevalence_diff, prevalence_theta_1, prevalence_theta_h_1 = [], [], []
    #     for b, bt in enumerate(beta):
    #         prevalence1 = np.load(path.join(res_path, 'beta1_{0:.3f}-beta2_{0:.3f}-theta_{1:.1f}'.format(bt, 1.0),
    #                                         'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
    #         prevalence2 = np.load(path.join(res_path, 'beta1_{0:.3f}-beta2_{0:.3f}-theta_{1:.1f}'.format(bt, -1.0),
    #                                         'T_0.{0}-prevalence_averaged.npy'.format(len(ts))))
    #         prevalence_diff.append((prevalence1[-1] - prevalence2[-1]) / prevalence1[-1])
    #         prevalence_theta_1.append(prevalence1[-1] / h_tnet.n)
    #         prevalence_theta_h_1.append(prevalence2[-1] / h_tnet.n)
    #     ydata = prevalence_diff
    #     ax[0][0].plot(beta, ydata, c=COLORS[d], clip_on=False)
    # ax[0][0].tick_params(direction='in', length=2.2, width=0.7)
    # ax[0][0].set_ylim([-0.05, 0.7])
    # ax[0][0].set_xscale('log')
    # ax[0][0].set_xlabel(r'$\beta$')
    # ax[0][0].set_ylabel(r'$\frac{\rho(\Theta=1)-\rho(\Theta=d-1)}{\rho(\Theta=1)}$')

    h_tnet1 = hyperTN(datasets[5])  # plot face-to-face datasets
    ts1 = h_tnet1.time_division(which='all')
    for idtt, tt in enumerate([1.0, -1.0]):
        weights_sum, prev = [], []
        for bt in beta:
            backbone = h_tnet1.return_backbone({'beta': bt, 'theta': tt}, subnet=len(ts1) - 1)
            weights_sum.append([sum([backbone[k] for k in backbone if len(k) == od]) / h_tnet1.n for od in [2, 3]])
            prev.append(sum([backbone[k] for k in backbone]) / h_tnet1.n)
        for iod, od in enumerate([2, 3]):
            ax[0].plot(beta, [weights_sum[i][iod] / prev[i] for i in range(len(beta))],
                            linestyle=LINESTYLES[idtt], c=COLORS[iod], clip_on=False)
    ax[0].set_xscale('log')
    ax[0].tick_params(direction='in', length=2.2, width=0.7)
    ax[0].set_xlabel(r'$\beta$', fontsize=9)
    ax[0].set_ylabel(r'$\rho_d$', fontsize=9)
    ax[0].xaxis.set_tick_params(labelsize=6)
    ax[0].yaxis.set_tick_params(labelsize=6)

    metrics = ['linkweight', 'time_independent_link_local_metric_subplink_alpha1.0',
               'time_independent_link_local_metric_subplink_alpha-1.0',
               'time_independent_link_local_metric_adjplink_alpha1.0',
               'time_independent_link_local_metric_adjplink_alpha-1.0']
    metrics = metrics + ['time_dependent_link_local_metric_subplink_alpha1.0',
                         'time_dependent_link_local_metric_subplink_alpha-1.0',
                         'time_dependent_link_local_metric_adjplink_alpha1.0',
                         'time_dependent_link_local_metric_adjplink_alpha-1.0']
    corrs = []
    for itt, theta in enumerate([1.0, -1.0]):
        corr_coef = []
        for bt in beta:
            backbone1 = h_tnet1.return_backbone(params={'beta': bt, 'theta': theta}, subnet=len(ts1) - 1)
            corr_coef.append([corr_coef_backbone_metric(h_tnet1, {'beta': bt, 'theta': theta}, subnet=len(ts1) - 1,
                                                     order=3, metric=metric, corr_metric='kendalltau', coef=bt)
                           for metric in metrics])
        corrs.append(corr_coef)
    for m in range(len(metrics)):
        if 'time_dependent' in metrics[m]:
            ax[1].plot(beta, [e[m] for e in corrs[0]], '-', linewidth=1,
                            c=COLORS[m - 4], label=metrics[m], alpha=0.8)
            ax[2].plot(beta, [e[m] for e in corrs[1]], '-', linewidth=1,
                            c=COLORS[m - 4], label=metrics[m], alpha=0.8)
        else:
            ax[1].plot(beta, [e[m] for e in corrs[0]], ':', linewidth=1.4,
                            c=COLORS[m], label=metrics[m], alpha=0.8)
            ax[2].plot(beta, [e[m] for e in corrs[1]], ':', linewidth=1.4,
                            c=COLORS[m], label=metrics[m], alpha=0.8)

    ax[1].plot([0.0008, 1.1], [0, 0], '--', linewidth=0.6, c='grey')
    ax[2].plot([0.0008, 1.1], [0, 0], '--', linewidth=0.6, c='grey')
    ax[1].set_xlabel(r'$\beta$', fontsize=9)
    ax[2].set_xlabel(r'$\beta$', fontsize=9)
    ax[1].xaxis.set_tick_params(labelsize=6)
    ax[2].xaxis.set_tick_params(labelsize=6)
    ax[1].tick_params(direction='in', length=2.2, width=0.7)
    ax[2].tick_params(direction='in', length=2.2, width=0.7)
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')
    ax[1].set_ylim([-1.0, 1.0])
    ax[2].set_ylim([-1.0, 1.0])
    ax[1].set_ylabel(r'Kendall tau', fontsize=9)
    ax[2].set_ylabel(r'Kendall tau', fontsize=9)
    ax[1].set_yticks([-1, -0.5, 0, 0.5, 1.0], ['$-1.0$', '$-0.5$', '$0.0$', '$0.5$', '$1.0$'], fontsize=6)
    ax[2].set_yticks([-1, -0.5, 0, 0.5, 1.0], ['$-1.0$', '$-0.5$', '$0.0$', '$0.5$', '$1.0$'], fontsize=6)
    # handles, labels = ax[-1][-1].get_legend_handles_labels()
    # labels = [r'$w_j$', r'$\xi_j^{sub-pairwise}(\alpha=1)$', r'$\xi_j^{sub-pairwise}(\alpha=-1)$',
    #           r'$\xi_j^{adj-pairwise}(\alpha=1)$',
    #           r'$\xi_j^{adj-pairwise}(\alpha=-1)$',
    #           r'$\Xi_j^{sub-pairwise}(\alpha=1)$', r'$\Xi_j^{sub-pairwise}(\alpha=-1)$',
    #           r'$\Xi_j^{adj-pairwise}(\alpha=1)$',
    #           r'$\Xi_j^{adj-pairwise}(\alpha=-1)$']
    # fig.legend(handles, labels, fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4,
    #            columnspacing=0.5)

    ax[0].text(.97, .97, '(A)', ha='right', va='top', transform=ax[0].transAxes, fontsize='medium', fontweight='bold')
    ax[1].text(.97, .97, '(B)', ha='right', va='top', transform=ax[1].transAxes, fontsize='medium', fontweight='bold')
    ax[2].text(.97, .97, '(C)', ha='right', va='top', transform=ax[2].transAxes, fontsize='medium', fontweight='bold')
    fig.tight_layout()
    fig.savefig(path.join(PATH_TO_FIGS, 'NetSci2024_.pdf'), dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contation process on temporal higher-order networks')
    parser.add_argument('--dataset', type=str, default='infectious', help='dataset')
    parser.add_argument('--beta', type=float, default=0.25, help='infectivity for pairwise interaction')
    parser.add_argument('--theta', type=str, default='1/2',
                        help='threshold for contagion to occur in hyperlink interaction')
    parser.add_argument('--R', type=int, default=100, help='the number of realizations in total')
    parser.add_argument('--n_arrays', type=int, default=10, help='the number of job arrays in slurm')
    parser.add_argument('--array_id', type=int, default=1, help='ID of job arrays in slurm')
    args = parser.parse_args()
    h_tnet = hyperTN(args.dataset)

    # groupsize_statistics()
    # plot_prevalence(h_tnet)
    # prevalence_diff(xaxis='beta', yaxis='')
    # average_shuffled_outputs(args.dataset, args.beta, num_r=100, num_s=10, theta=args.theta)
    # integrate_backbones(h_tnet, beta=args.beta, theta=args.theta, num_r=5000)
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
    # compare_diff_metrics_corr(order=3, theta=args.theta, corr_metric='kendalltau', xaxis='beta')
    # compare_diff_exponent_coef_metrics_corr(theta=args.theta, order=3, corr_metric='kendalltau')
    # compare_time_decayed_metrics_corr(order=3, theta=args.theta, corr_metric='kendalltau')
    # compare_relative_weight(xvar='beta', normalize=True)
    # colormap2d(h_tnet, h_tnet.time_division(which='largest')[0])
    corr_diff_theta(spearmanr)
    # distribution_log_ratio_weights(h_tnet, order=3)
    # prevalence_comparison()
    # netsci_fig()
