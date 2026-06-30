import os
from random import sample
import numpy as np
import json
from collections import Counter
from config import PATH_TO_RESULTS, PATH_TO_NETWORK_FILE
# from utils import *


class HyperTN:
    def __init__(self, dataname):
        self.dataname = dataname
        self.hypercontacts = self._hypercontacts_from_txt(os.path.join(PATH_TO_NETWORK_FILE, self.dataname + '_hypergraph.dat'))
        node_set = set()
        for contacts in self.hypercontacts:
            for contact in contacts:
                node_set.update(contact)
        minid, maxid = min(node_set), max(node_set)
        assert minid == 0 and maxid == len(node_set) - 1
        self.n = maxid + 1
        self.T = len(self.hypercontacts)

        self._print_info()

    def _print_info(self):
        n_events = sum([len(e) for e in self.hypercontacts])
        print(f'---- Temporal hypergraph information ----\n'
              f'Name: {self.dataname}\n#nodes: {self.n}\t#hyperlinks: {len(self.aggregate(self.hypercontacts))}\t#hyperevents: {n_events}\ttime span: [0, {self.T}]\n'
              f'-------------------------------------')

    def _hypercontacts_from_txt(self, fpath: str) -> list:
        '''
        Load hypercontacts from txt file.
        :param fpath: file path to load the hypercontacts.
        :return: list of hypercontacts, each element is a list of hypercontacts occured at a specific timestamp.
        '''
        hyperlinks = []
        with open(fpath, 'r') as f:
            line = f.readline()
            while line:
                hyperlinks_t = json.loads(line.strip())
                hyperlinks.append([frozenset(e) for e in hyperlinks_t])
                line = f.readline()

        return hyperlinks

    def hypercontacts2txt(self, fpath):
        '''
        Save hypercontacts into txt file.
        :param fpath: file path to save the hypercontacts.
        '''
        with open(fpath+'/{0}'.format(self.dataname+'_hypergraph.dat'), 'a') as f:
            for hypercontacts_one_snapshot in self.hypercontacts:
                f.write(json.dumps([list(e) for e in hypercontacts_one_snapshot])+'\n')

    def aggregate(self, hcontacts: list) -> Counter:
        '''
        Aggregate temporal hypergraph along time axis.
        :param hypercontacts: each element is a list of hypercontacts occured at a specific timestamp.
        :return: A Counter object with key being hlink and count being the link weight.
        '''
        hcontacts = list(map(lambda l: [frozenset(s) for s in l], hcontacts))
        cnt = Counter()
        for hedges in hcontacts:
            cnt.update(hedges)

        return cnt

    def simulate_threshold_model(self, seedset: frozenset, params: dict, T):
        '''
        The threshold model on temporal hypergraphs. Adapted from ref. Social contagion models on hypergraphs. Guilherme et al. Phys. Rev. Res., 2(2):023032, 2020
        For a hyper-contact with size 2, the directed infection occurs between the susceptible node and its infectious neighbor, beta1
        For a hyper-contact with size > 2: when the number of infectious nodes in the hyper-edge is greater than theta * N,
        each susceptible node within it has a prob. beta2 to be infected.
        :param seedset: the seed node (or set in a more general case) to initiate the process.
        :param params: model parameters, including 'beta1', 'beta2', and 'theta'.
        :param T: total time steps.
        :return: backbone and prevalence
        '''
        beta, theta = [params[k] for k in ['beta', 'theta']]
        assert theta == '1' or theta == 'd-1'

        infected_now = set(seedset)  # infected nodes at the current timestamp
        new_infected_t = set()  # newly infected nodes during current timestamp
        n_infected = np.zeros(T, dtype=np.int32)  # prevalence as a function of t.
        new_infected_hlinks = dict()  # newly infected nodes (keys) and the links (values) used for infections.
        diffusion_links = Counter()  # a list of links upon which the diffusion occurred.
        for t, hlinks in enumerate(self.hypercontacts):  # at time t in [0, T) , iterate all hlinks.
            if t >= T:
                break
            for hlink in hlinks:
                infected_nodes = infected_now.intersection(hlink)  # infected nodes at the timestamp before t
                threshold = 1 if theta == '1' else len(hlink) - 1

                if len(infected_nodes) >= threshold:
                    for node in hlink:
                        if node in infected_nodes:  # already infected before t
                            continue
                        elif np.random.uniform() <= beta:
                            if not node in new_infected_hlinks:
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

        return diffusion_links, n_infected

    def load_backbone(self, params: dict, n_realizations = 50000):
        '''
        Read the backbone from pickle file for given parameters, and do normalization by the number of realizations .
        :param params: model parameters, including 'beta' and 'theta'.
        :param n_realizations: number of realizations used to generate the backbone.
        :return: backbone as a dictionary with key being hyperlink and value being the normalized weight.
        '''
        import pickle

        beta, theta = params['beta'], params['theta']

        pkl_path = os.path.join(PATH_TO_RESULTS, self.dataname, 'threshold_model', 'beta_{0:.3f}-theta_{1}'.format(beta, theta), 'T_0.9-backbone.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError('Backbone file does not exist!', 'beta_{0:.3f}-theta_{1}'.format(beta, theta))
        
        backbone = dict()
        with open(pkl_path, 'rb') as f:
            backbone = pickle.load(f)
        backbone_normalized = {k: v / n_realizations for k, v in backbone.items()}

        return backbone_normalized

    def integrate_backbones(self, beta, theta, num_r=1000):
        '''
        Integrate backbones from multiple realizations stored in pickle files, and save the re.
        :param beta: infectivity parameter.
        :param theta: threshold parameter.
        :param num_r: number of realizations.
        '''
        import pickle

        res_path = os.path.join(PATH_TO_RESULTS, self.dataname, 'threshold_model', 'beta_{0:.3f}-theta_{1}'.format(beta, theta))

        backbone = Counter()
        for r in range(1, num_r + 1):
            with open(os.path.join(res_path, 'backbone-R{0}.pkl'.format(r)), 'rb') as f:
                bb = pickle.load(f)
                backbone.update(bb)

        with open(os.path.join(res_path, 'backbone.pkl'), 'wb') as f:
            pickle.dump(backbone, f)
