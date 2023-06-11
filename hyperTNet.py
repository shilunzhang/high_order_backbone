import os
import numpy as np
import json
from collections import Counter
from functools import reduce
import matplotlib.pyplot as plt
from TNet import *


class hyperTN:
    def __init__(self, tnet: TN):
        self.tnet = tnet
        self.dataname = tnet.dataname
        self.hypercontacts = self.hypercontacts_from_txt(tnet.pairwise2hyperlink())

    def hypercontacts_from_txt(self, fpath: str) -> list:
        hyperlinks = []
        with open(fpath, 'r') as f:
            line = f.readline()
            while line:
                hyperlinks_t = json.loads(line.strip())
                hyperlinks.append([frozenset(e) for e in hyperlinks_t])
                line = f.readline()

        return hyperlinks

    # TODO: aggregate temporal hypergraph along temporal axis
    def aggregate_hyperTN(self, hcontacts: list) -> Counter:
        '''
        Aggregate temporal hypergraph along time axis.
        :param hypercontacts: each element is a list of hypercontacts occured at a specific timestamp.
        :return: list of hyperedges of the resultant aggregated hypergraph
        '''
        hcontacts = list(map(lambda l: [frozenset(s) for s in l], hcontacts))
        cnt = Counter()
        for hedges in hcontacts:
            cnt.update(hedges)

        return cnt

    def threshold_model(self, seedset: frozenset, params: dict, T):
        '''
        The threshold model on temporal hyper graphs. Adapted from ref. Social contagion models on hypergraphs. Guilherme et al. Phys. Rev. Res., 2(2):023032, 2020
        For a hyper-contact with size 2, the directed infection occurs between the susceptible node and its infectious neighbor, beta1
        For a hyper-contact with size > 2, the threshold process comes into effect: when the number of infectious nodes in the hyper-edge is greater than theta * N,
        the susceptible nodes has a prob. to be infected, beta2.
        In the simplest case, beta1 == 1, beta == 1. (start point)
        :param seedset:
        :param beta1:
        :param theta:
        :return: diffusion backbone and prevalence
        '''
        beta1, beta2, theta = [params[k] for k in ['beta1', 'beta2', 'theta']]
        infected_now = set(seedset)
        infected_t = set()
        n_infected = np.zeros(T, dtype=np.int32)
        diffusion_links = []  # a list of links upon which the diffusion, starting from the seedset, occurred.
        for t, hlinks in enumerate(self.hypercontacts):
            # print(hlinks)
            if t >= T:
                break
            for id, hlink in enumerate(hlinks):
                if len(hlink) == 2:
                    infected_nodes = infected_now.intersection(hlink)
                    if len(infected_nodes) == 1 and np.random.uniform() <= beta1:
                        infected_t.update(hlink)
                        diffusion_links.append(hlink)
                elif len(hlink) > 2:
                    infected_nodes = infected_now.intersection(hlink)
                    if len(infected_nodes) >= theta:
                        for node in hlink:
                            if node in infected_nodes:
                                continue
                            elif np.random.uniform() <= beta2:
                                infected_t.add(node)
                                diffusion_links.append(hlink)
                else:
                    print('Something wrong the hlink size..')
            infected_now.update(infected_t)
            n_infected[t] = len(infected_now)
            infected_t.clear()

        return diffusion_links, n_infected

    def simplicial_contagion(self, seedset, infectivity: tuple, order=2):  # TODO: simplicial contagion model
        '''
        Simplicial contagion SI model in higher-order temporal networks. Adapted from ref. Simplicial contagion model in temporal higher-order networks.
        :param infectivity_params: infection probabilities in different orders.
        :param order: the highest order considered.
        :return:
        '''

        infected_now = set(seedset)
        infected_t = set()
        n_infected = np.zeros(len(self.hypercontacts), dtype=np.int32)
        hlink_id = [set() for _ in self.hypercontacts]
        for t, hlinks in enumerate(self.hypercontacts):
            # print(hlinks)
            for id, hlink in enumerate(hlinks):
                if len(hlink) == 2:
                    infected_nodes = infected_now.intersection(hlink)
                    if len(infected_nodes) == 1 and np.random.uniform() <= beta1:
                        infected_t.update(hlink)
                        hlink_id[t].add(id)
                elif len(hlink) > 2:
                    infected_nodes = infected_now.intersection(hlink)
                    if len(infected_nodes) >= theta:
                        for node in hlink:
                            if node in infected_nodes:
                                continue
                            elif np.random.uniform() <= beta2:
                                infected_t.add(node)
                                hlink_id[t].add(id)
                else:
                    print('Something wrong the hlink size..')
            infected_now.update(infected_t)
            n_infected[t] = len(infected_now)
            # if t < 50:
            #     print(t, infected_now)
            infected_t.clear()

        return hlink_id, n_infected


#TODO: the nonlinear kernel when there are multiple interactions.
def nonlinear_model(seed: frozenset):
    pass


if __name__ == '__main__':
    dname = 'infectious'
    tnet = TN(dname)
    hyper_tnet = hyperTN(tnet)

    # spread_diff_seeds(hyper_tnet, {'beta1': 1.0, 'beta2': 1.0, 'theta': 2})
    # print(groupsize_statistics(hyper_tnet))