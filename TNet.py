import os
import os.path as path
from platform import system
import json
import numpy as np

CURRENT_SYSTEM = system()
if CURRENT_SYSTEM == 'Windows':
    PATH_HOME = path.join('D:\\', 'high_order_backbone')
elif CURRENT_SYSTEM == 'Linux':
    PATH_HOME = path.join('/tudelft.net', 'staff-bulk', 'ewi', 'insy', 'MMC', 'shilunzhang', 'high_order_backbone')
else:
    raise ValueError('system(): ', CURRENT_SYSTEM, ', neither Windows nor Linux.')
PATH_TO_NETWORK_FILE = path.join(PATH_HOME, 'datasets')
PATH_TO_RESULTS = path.join(PATH_HOME, 'results')
PATH_TO_FIGS = path.join(PATH_HOME, 'figs')

TN_resolution = {'hs11': 20, 'hs12': 20, 'work1': 20, 'work2': 20, 'ht09': 1, 'test': 1, 'test1': 1}


class TN:
    def __init__(self, dname, resolution=1, ret=True, datapath=PATH_TO_NETWORK_FILE):
        self.dataname = dname
        self.datapath = datapath
        self.resolution = resolution
        self.contacts = self._read_TN(datapath)
        self.nodes = np.unique(self.contacts[:, :2])  # individual's ID
        self.n = len(self.nodes)  # number of individuals
        self.startT = min(self.contacts[:, 2])
        self.endT = max(self.contacts[:, 2])
        self.T = self.endT - self.startT + 1
        self.info = {'ret': ''}
        if ret:
            self._remove_empty_timestamps()
        assert self.startT == 0

        # adjacency matrix of aggregated network
        self.backbone = np.zeros([self.n, self.n], dtype=int)  # weighted adjacency matrix of time-aggregated network.
        for p1, p2, t in self.contacts:
            self.backbone[p1, p2] += 1
            self.backbone[p2, p1] += 1

        self.print_info()

    def _read_TN(self, data_path):
        # read data and compute the time step according to the resolution.
        data_arr = np.loadtxt(path.join(data_path, f'{self.dataname}.dat'), dtype=np.int32)  # contacts record.
        if max(data_arr[:, 0]) > 5e4:
            data_arr = data_arr[:, [1, 2, 0]]
        data_arr[:, 2] = data_arr[:, 2] // self.resolution  # one time unit denotes the resolution.

        return self._relabel_id(data_arr)

    def _remove_duplicated_contacts(self, allow_selfloop=False):
        # remove multiple contacts between two nodes in a single timestamp
        contact_m = np.diag([~allow_selfloop] * self.n)
        res_idx = np.ones(len(self.contacts), dtype=bool)
        i = 0
        tt = 0
        while i < len(self.contacts):
            n1, n2, t = self.contacts[i]
            if t == tt:
                if not contact_m[n1][n2]:
                    contact_m[n1][n2] = True
                    contact_m[n2][n1] = True
                else:
                    # print('remove ', n1, n2, t, 'at line ', i)
                    res_idx[i] = False
                i += 1
            else:
                contact_m = np.diag([~allow_selfloop] * self.n)
                tt = t
        self.contacts = self.contacts[res_idx]
        print('Deleted {0} contacts (selfloops) in single time step. It contains {1} contacts after deletion'.format(
            np.sum(~res_idx), len(self.contacts)))

        return self.contacts

    def _remove_empty_timestamps(self, replace=True, write_txt=False):
        t_now, t_temp = 0, 0
        new_ts = -np.ones(self.contacts.shape[0])  # initialize new constructed timestamps.
        for i in range(self.contacts.shape[0]):
            node1, node2, t = self.contacts[i, :]
            new_ts[i] = t_now
            if t > t_temp:
                t_now += 1
                new_ts[i] = t_now
                t_temp = t

        if replace:
            self.contacts[:, 2] = new_ts
            self.info['ret'] = '_ret'
            self.startT = 0
            self.endT = max(self.contacts[:, 2])
            self.T = self.endT - self.startT + 1

        if write_txt:  # save as text file with .ret file extension.
            np.savetxt(path.join(self.datapath, f'{self.dataname}.ret'), self.contacts, fmt='%d\t%d\t%d')

        return new_ts

    def _relabel_id(self, contacts=None, start_id=0):
        # relabel node id according to the appearance time.
        if contacts is None:
            contacts = self.contacts
        node_map = dict()
        id = start_id
        for i, (node1, node2, t) in enumerate(contacts):
            if node1 not in node_map:
                node_map[node1] = id
                id += 1
            if node2 not in node_map:
                node_map[node2] = id
                id += 1
            contacts[i] = [node_map[node1], node_map[node2], t]

        return contacts

    def _iterate(self, i=0):  # create periodic temporal pattern.
        self.contacts[:, 2] -= min(self.contacts[:, 2])
        dataset_ = self.contacts.copy()
        for _ in range(i):
            dataset_[:, 2] += self.T
            self.contacts = np.concatenate((self.contacts, dataset_))

        return self.contacts

    def print_info(self):
        print(f'---- Temporal netwok information ----\n'
              f'Name: {self.dataname}\n#nodes: {self.n}\t#contacts: {len(self.contacts)}\ttime span: [{self.startT}, {self.endT}]\n'
              f'-------------------------------------')

    def node_birthtime(self):
        birth_time = -np.ones(self.nodes.shape, dtype=np.int32)
        for p1, p2, t in self.contacts:
            if birth_time[p1] < 0:
                birth_time[p1] = t
            if birth_time[p2] < 0:
                birth_time[p2] = t
        assert np.all(birth_time >= 0), print('Birth time of some nodes are negative.')
        return birth_time

    def node_strength(self, weighted=True):
        adj_mat = self.backbone.copy()
        if not weighted:
            adj_mat = adj_mat.astype(bool)

        return adj_mat.sum(axis=0)

    def pairwise2hyperlink(self) -> str:  # construct temporal hypergraph from temporal pairwise interactions.
        '''

        :param tnet:
        :return: file path as string type.
        '''
        path_res = path.join(PATH_TO_RESULTS, self.dataname, 'hyperlinks')

        if not path.exists(path_res):
            os.mkdir(path_res)

        file_path = path.join(path_res, self.dataname + '_hypergraph.dat')
        if not path.exists(file_path):
            f = open(file_path, 'w+')  # TODO check if already exists, if so delete it first
            l = 0
            ll = 0  #
            tlast = self.startT
            n_events = 0
            while l <= len(self.contacts):
                # n1, n2, t = tnet.contacts[l]
                if l == len(self.contacts) or self.contacts[l, 2] != tlast:
                    print('t-step: ', tlast)
                    hlinks = [e for e in list(maximal_cliques_BK(self.n, self.contacts[ll:l, :2])) if len(e) >= 2]
                    # print(type(hlinks[0][0]))
                    f.write(json.dumps(hlinks) + '\n')
                    n_events += len(hlinks)
                    if l < len(self.contacts):
                        ll = l
                        tlast = self.contacts[l, 2]
                    else:
                        print('Number of events: ', n_events)
                        break
                elif self.contacts[l, 2] == tlast:
                    l += 1

            f.close()

        return file_path


def maximal_cliques_BK(N, links):
    nn_dict = {node: set() for node in range(N)}
    for node1, node2 in links:
        nn_dict[node1].add(int(node2))
        nn_dict[node2].add(int(node1))

    if len(links) == 0:
        return

    Q = [None]
    cand = set(range(N))
    subg = cand.copy()
    stack = []

    u = max(subg, key=lambda u: len(cand & nn_dict[u]))  # pivot node
    ext_u = cand - nn_dict[u]

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                nn_q = nn_dict[q]
                subg_q = subg & nn_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & nn_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & nn_dict[u]))
                        ext_u = cand - nn_dict[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass



if __name__ == '__main__':
    gname = 'primaryschool'
    tnet = TN(gname, resolution=1)
    # aggregate_hyperTN(tnet.pairwise2hyperlink())
    import networkx as nx
    import matplotlib.pyplot as plt
    # g = nx.barbell_graph(5, 9)
    # print(g.nodes(), g.edges())
    # print(list(nx.find_cliques(g)))
    # print(list(maximal_cliques_BK(len(g), g.edges())))
    # nx.draw_networkx(g)
    # plt.show()
