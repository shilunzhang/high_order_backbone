import os
from random import sample
import numpy as np
import json
from collections import Counter
from functools import reduce
import matplotlib.pyplot as plt
from TNet import *
from utils import *


class hyperTN:
    def __init__(self, dataname):
        if dataname in {'infectious', 'primaryschool', 'highschool2013', 'ht09'}:
            self.datatype = 'phy-contact'
        elif dataname in {'q-bio', 'q-fin', 'quant-ph', 'nucl-th', 'hep-lat'}:
            self.datatype = 'sci-collaboration'
        else:
            print('dataset does not exist..')
        self.dataname = dataname
        if self.datatype == 'phy-contact':
            tnet = TN(self.dataname)
            self.hypercontacts = self.hypercontacts_from_txt(tnet.pairwise2hyperlink())
            self.n = tnet.n
        else:
            self.hypercontacts = self.hypercontacts_from_txt(path.join(PATH_TO_RESULTS, self.dataname, 'hyperlinks', self.dataname + '_hypergraph.dat'))
            minid, maxid = 1e5, 0
            for contacts in self.hypercontacts:
                for contact in contacts:
                    minid_contact, maxid_contact = min(contact), max(contact)
                    if maxid_contact > maxid:
                        maxid = maxid_contact
                    if minid_contact < minid:
                        minid = minid_contact
            assert minid == 0
            self.n = maxid
        self.T = len(self.hypercontacts)

        self.print_info()

    def print_info(self):
        n_events = sum([len(e) for e in self.hypercontacts])
        print(f'---- Temporal hypergraph information ----\n'
              f'Name: {self.dataname}\n#nodes: {self.n}\t#events: {n_events}\ttime span: [0, {self.T}]\n'
              f'-------------------------------------')
    def hypercontacts_from_txt(self, fpath: str) -> list:
        hyperlinks = []
        with open(fpath, 'r') as f:
            line = f.readline()
            while line:
                hyperlinks_t = json.loads(line.strip())
                hyperlinks.append([frozenset(e) for e in hyperlinks_t])
                line = f.readline()

        return hyperlinks

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

    def shuffle_order_within_snapshot(self, seed=0, r=10) -> str:
        path_res = path.join(PATH_TO_RESULTS, self.dataname)
        file_path = path.join(path_res, 'hyperlinks', 'hyperTN_shuffled_IDs.json')
        if os.path.exists(file_path):
            return file_path

        rng = np.random.default_rng(seed)
        f = open(file_path, 'w+')
        for hlinks in self.hypercontacts:
            id_t = []
            for i in range(r):
                id_t.append(rng.choice(np.arange(len(hlinks)), size=len(hlinks), replace=False).tolist())
            f.write(json.dumps(id_t)+'\n')
        f.close()

        return file_path

    def SI_model(self, seedset: frozenset, beta, T):
        '''
        SI like process on temporal hypergraphs.
        :param seedset: infected nodes initially
        :param beta: infection probability per time step
        :param T:
        :return:
        '''
        infected_now = set(seedset)
        infected_t = set()
        n_infected = np.zeros(T, dtype=np.int32)
        diffusion_links = []
        for t, hlinks in enumerate(self.hypercontacts):
            if t >= T:
                break
            for id, hlink in enumerate(hlinks):
                pass

    def threshold_model_v0(self, seedset: frozenset, params: dict, T):
        '''
        The threshold model on temporal hyper graphs. Adapted from ref. Social contagion models on hypergraphs. Guilherme et al. Phys. Rev. Res., 2(2):023032, 2020
        For a hyper-contact with size 2, the directed infection occurs between the susceptible node and its infectious neighbor, beta1
        For a hyper-contact with size > 2, the threshold process comes into effect: when the number of infectious nodes in the hyper-edge is greater than theta * N,
        the susceptible nodes has a prob. to be infected, beta2.
        In the deterministic case, beta1 == 1, beta2 == 1. (start point), theta == 1 or s-1, i.e., all other nodes are infected except the target node.
        :param seedset:
        :return: diffusion backbone and prevalence
        '''
        beta1, beta2, theta = [params[k] for k in ['beta1', 'beta2', 'theta']]
        infected_now = set(seedset)
        new_infected_t = set()
        n_infected = np.zeros(T, dtype=np.int32)  # prevalence as a function of t.
        diffusion_links = Counter()  # a list of links upon which the diffusion occurred.
        # rng_shuffle = np.random.default_rng(0)  # random generator for randm shuffling of link orders.
        for t, hlinks in enumerate(self.hypercontacts):  # at time t, go over all hlinks at the time.
            if t >= T:
                break
            # for i in range(1):  # shuffle the order of hlinks
            #     id_shuffled = rng_shuffle.choice(np.arange(len(hlinks)), size=len(hlinks), replace=False)
            for id, hlink in enumerate(hlinks):
                assert isinstance(hlink, frozenset)
                # hlink = hlinks[id]
                if len(hlink) == 2:
                    infected_nodes = infected_now.intersection(hlink)
                    if len(infected_nodes) == 1 and np.random.uniform() <= beta1:
                        new_infected_t.update(hlink-infected_nodes)
                        diffusion_links.update([hlink])
                elif len(hlink) > 2:
                    diffused = 0
                    threhsold = len(hlink)-1 if theta < 0 else theta  #
                    infected_nodes = infected_now.intersection(hlink)
                    if len(infected_nodes) >= threhsold:
                        for node in hlink:
                            if node in infected_nodes:
                                continue
                            elif np.random.uniform() <= beta2:
                                new_infected_t.add(node)
                                diffused += 1
                    if diffused > 0:  # record the hlink when at least one node got infected. The added value is the number of node infected.
                        diffusion_links.update(Counter({hlink: diffused}))
            infected_now.update(new_infected_t)
            n_infected[t] = len(infected_now)
            new_infected_t.clear()

        return diffusion_links, n_infected

    def threshold_model_v1(self, shuffled_r, seedset: frozenset, params: dict, T):
        '''
        The threshold model on temporal hyper graphs. Adapted from ref. Social contagion models on hypergraphs. Guilherme et al. Phys. Rev. Res., 2(2):023032, 2020
        For a hyper-contact with size 2, the directed infection occurs between the susceptible node and its infectious neighbor, beta1
        For a hyper-contact with size > 2, the threshold process comes into effect: when the number of infectious nodes in the hyper-edge is greater than theta * N,
        the susceptible nodes has a prob. to be infected, beta2.
        In the deterministic case, beta1 == 1, beta2 == 1. (start point), theta == 1 or s-1, i.e., all other nodes are infected except the target node.
        :param seedset:
        :return: diffusion backbone and prevalence
        '''
        shuffled_IDs_path = self.shuffle_order_within_snapshot()
        shuffled_IDs = []
        with open(shuffled_IDs_path, 'r') as f:
            line = f.readline()
            while line:
                shuffled_IDs.append(json.loads(line.strip())[shuffled_r-1])
                line = f.readline()
        beta1, beta2, theta = [params[k] for k in ['beta1', 'beta2', 'theta']]
        infected_now = set(seedset)  # infected nodes at each timestamp
        new_infected_t = set()  # newly infected nodes in a timestamp
        n_infected = np.zeros(T, dtype=np.int32)  # prevalence as a function of t.
        diffusion_links = Counter()  # a list of links upon which the diffusion occurred.
        for t, hlinks in enumerate(self.hypercontacts):  # at time t, go over all hlinks in a shuffled order.
            if t >= T:
                break
            for id in range(len(hlinks)):
                hlink = hlinks[shuffled_IDs[t][id]]  # get a hlink in a shuffled order.
                assert isinstance(hlink, frozenset)
                infected_nodes = infected_now.intersection(hlink)  # infected nodes at the timestamp before t
                if len(hlink) == 2:
                    if len(infected_nodes) == 1 and len(new_infected_t.intersection(hlink-infected_nodes)) == 0 and np.random.uniform() <= beta1:
                        new_infected_t.update(hlink-infected_nodes)
                        diffusion_links.update([hlink])
                elif len(hlink) > 2:
                    diffused = 0
                    threshold = theta  #
                    if np.abs(theta + 1) < 1e-3:
                        threshold = len(hlink) - 1
                    elif np.abs(theta - 0.5) < 1e-3:
                        threshold = np.ceil(len(hlink)*0.5)
                    if len(infected_nodes) >= threshold:
                        for node in hlink:
                            if node in infected_nodes:
                                continue
                            elif node not in new_infected_t and np.random.uniform() <= beta2:
                                new_infected_t.add(node)
                                diffused += 1
                    if diffused > 0:  # record the hlink when at least one node got infected. The added value is the number of node infected.
                        diffusion_links.update(Counter({hlink: diffused}))
            infected_now.update(new_infected_t)
            n_infected[t] = len(infected_now)
            new_infected_t.clear()
            # assert counter_total(diffusion_links) == n_infected[t]-len(seedset)

        return diffusion_links, n_infected

    def threshold_model(self, seedset: frozenset, params: dict, T):
        '''
        The threshold model on temporal hyper graphs. Adapted from ref. Social contagion models on hypergraphs. Guilherme et al. Phys. Rev. Res., 2(2):023032, 2020
        For a hyper-contact with size 2, the directed infection occurs between the susceptible node and its infectious neighbor, beta1
        For a hyper-contact with size > 2, the threshold process comes into effect: when the number of infectious nodes in the hyper-edge is greater than theta * N,
        the susceptible nodes has a prob. to be infected, beta2.
        In the deterministic case, beta1 == 1, beta2 == 1. (start point), theta == 1 or s-1, i.e., all other nodes are infected except the target node.
        :param seedset:
        :return: diffusion backbone and prevalence
        '''
        beta1, beta2, theta = [params[k] for k in ['beta1', 'beta2', 'theta']]
        assert beta1 == beta2
        beta = beta1
        infected_now = set(seedset)  # infected nodes at each timestamp
        new_infected_t = set()  # newly infected nodes in a timestamp
        n_infected = np.zeros(T, dtype=np.int32)  # prevalence as a function of t.
        new_infected_hlinks = dict()  # newly infected nodes (keys) and the links (values) used for infections.
        diffusion_links = Counter()  # a list of links upon which the diffusion occurred.
        for t, hlinks in enumerate(self.hypercontacts):  # at time t, go over all hlinks in a shuffled order.
            if t >= T:
                break
            for hlink in hlinks:
                infected_nodes = infected_now.intersection(hlink)  # infected nodes at the timestamp before t
                threshold = theta  #
                if np.abs(theta + 1) < 1e-3:  # threshold == n - 1
                    threshold = len(hlink) - 1
                elif np.abs(theta - 0.5) < 1e-3:  # threshold >= 0.5 * n
                    threshold = np.ceil(len(hlink)*0.5)
                if len(infected_nodes) >= threshold:
                    for node in hlink:
                        if node in infected_nodes:
                            continue
                        elif np.random.uniform() <= beta:
                            if node not in new_infected_hlinks:
                                new_infected_t.add(node)
                                new_infected_hlinks[node] = []
                            new_infected_hlinks[node].append(hlink)
            for node in new_infected_hlinks:
                hlink_diffused = sample(new_infected_hlinks[node], k=1)
                diffusion_links.update(hlink_diffused)
            infected_now.update(new_infected_t)
            n_infected[t] = len(infected_now)
            new_infected_t.clear()
            new_infected_hlinks.clear()
            # assert counter_total(diffusion_links) == n_infected[t]-len(seedset)

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

    # def foremost_paths(self, source, t_start, t_end):
    #     t_foremost = 10 * self.T * np.ones(self.n, dtype=np.int32)  # the foremost time to reach a node
    #     p_foremost = [[] for i in range(tnet.n)]  # The unique foremost paths that reach a target node.
    #     t_foremost[source] = t_start - 1  # the source node is already reached.
    #     for t, events in enumerate(self.hypercontacts):
    #         if t >= t_end:
    #             break
    #         elif t < t_start:
    #             continue
    #         for event in events:
    #             for node in event:
    #             if t > t_foremost[node1]:  # if node1 has already been reached before t
    #                 if t > t_foremost[node2]:  # node2's foremost paths already found before
    #                     continue
    #                 elif t == t_foremost[node2]:  # more than one path, add path
    #                     p_foremost[node2].append((node1, node2, t))
    #                 elif t < t_foremost[node2]:  # the foremost path is found
    #                     t_foremost[node2] = t
    #                     p_foremost[node2] = [(node1, node2, t)]
    #             elif t > t_foremost[node2]:  # another direction of the contact
    #                 if t == t_foremost[node1]:
    #                     p_foremost[node1].append((node1, node2, t))
    #                 else:
    #                     t_foremost[node1] = t
    #                     p_foremost[node1] = [(node1, node2, t)]
    #
    #     return return_foremost_paths(t_foremost, p_foremost)


#TODO: the nonlinear kernel when there are multiple interactions.
def nonlinear_model(seed: frozenset):
    pass

if __name__ == '__main__':
    dname = 'ht09'
    tnet = TN(dname)
    hyper_tnet = hyperTN(tnet)
    hyper_tnet.shuffle_order_within_snapshot()
    # spread_diff_seeds(hyper_tnet, {'beta1': 1.0, 'beta2': 1.0, 'theta': hyper_tnet.n})
    # print(groupsize_statistics(hyper_tnet))
