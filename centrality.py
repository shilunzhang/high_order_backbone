from copy import copy

import numpy as np

from hyperTNet import *


def sublink_weights_order3(h_tnet: hyperTN, i=0) -> dict:
    '''
    Calculate the metric of the 3rd order link based on its weight w_j and the weights of its sub-links as w_j*(w1+w2+w3)
    :param h_tnet: temporal hypergraph
    :param i: the sub-net index
    :return:
    '''
    from itertools import combinations
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
    hlinks_order3_metric = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 3], 0)
    for hlink in hlinks_order3_metric:
        for hl in combinations(hlink, 2):
            hl = frozenset(hl)
            if hl in agg_hyperlinks:
                hlinks_order3_metric[hlink] += agg_hyperlinks[hlink] * agg_hyperlinks[hl]

    return hlinks_order3_metric


def effective_weights_order3(h_tnet: hyperTN, inverse=False, i=0) -> dict:
    '''
    Calculate the effective weight product of the 3rd order link based on its weight w_j and the weights of its sub-links
    :param h_tnet: temporal hypergraph
    :param inverse: inverse the weight sum of sub-links
    :param i: the sub-net index
    :return:
    '''
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
    hlinks_order3_metric = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 3], 0)
    hlinks_order3_count_tmp = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 3], 0)  # record if hlink3 appears
    hlinks_order3_subsetcount = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 3],
                                                0)  # record weights of sub-links
    for hlinks in h_tnet.hypercontacts[:ts[i]]:
        for hlink in hlinks:
            if len(hlink) == 2:
                for hlink3 in hlinks_order3_metric:
                    if hlink.issubset(hlink3):
                        hlinks_order3_subsetcount[hlink3] += 1
            elif len(hlink) == 3:
                hlinks_order3_count_tmp[hlink] += 1
        for hlink3 in hlinks_order3_metric:
            if hlinks_order3_count_tmp[hlink3] > 0:
                hlinks_order3_metric[hlink3] += (
                    hlinks_order3_subsetcount[hlink3] if not inverse else 1 / (1 + hlinks_order3_subsetcount[hlink3]))
                hlinks_order3_count_tmp[hlink3] = 0

    return hlinks_order3_metric


def local_effective_weights_(h_tnet: hyperTN, i=0) -> dict:
    '''
    calculate the effective weights for arbitrary order hyperlinks
    :param h_tnet:
    :param i:
    :return: a dictionary containing the effective weights for all hyperlinks
    '''
    import queue
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
    hlinks_order2_weights_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2],
                                              0)  # pairwise link weight at time t
    hlinks_metric = dict().fromkeys([k for k in agg_hyperlinks], 0)

    hlinks_superset = dict().fromkeys([k for k in agg_hyperlinks])
    hlinks_subset = dict().fromkeys([k for k in agg_hyperlinks if len(k) > 2])
    hlinks_higher_order_timestamp = dict().fromkeys([k for k in agg_hyperlinks if len(k) > 2])
    for k in agg_hyperlinks:
        hlinks_superset[k] = []
        if len(k) > 2:
            hlinks_subset[k] = []
        else:
            continue
        hlinks_higher_order_timestamp[k] = []

    for hlink in agg_hyperlinks:  # construct a superset and subset.
        for hlink_higher_order in hlinks_higher_order_timestamp:
            if len(hlink) + 1 == len(hlink_higher_order):
                remaining_nodes = hlink_higher_order - hlink
                if len(remaining_nodes) == 1:
                    hlinks_superset[hlink].append(list(remaining_nodes)[0])
                    hlinks_subset[hlink_higher_order].append(list(remaining_nodes)[0])

    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[i]]):  # record the timestamps of higher-order links
        for hlink in hlinks:
            if len(hlink) > 2:
                continue
            hlinks_higher_order_timestamp[hlink].append(t)

    q = queue.Queue()
    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[i]]):  # for each pairwise link, calculate score along the tree
        for hlink in hlinks:
            if len(hlink) == 2:
                q.put(hlink)
                while not q.empty():
                    hlink_current = q.get()
                    for node in hlinks_superset[hlink_current]:
                        hlink_next = hlink_current.union({node})
                        q.put(hlink_next)
                        # update hlink_next's score

            hlinks_order2_weights_t[hlink] += 1

    return hlinks_metric


def local_effective_weights(h_tnet: hyperTN, i=0) -> dict:
    '''
    calculate the effective weights for arbitrary order hyperlinks using a tree structure
    :param h_tnet:
    :param i:
    :return: a dictionary containing the effective weights for all hyperlinks
    '''
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])

    metrics = dict().fromkeys([k for k in agg_hyperlinks], 0)

    hlinks_superset = dict().fromkeys([k for k in agg_hyperlinks])  # record parental hyperlinks for each link
    hlinks_ho_timestamps = dict().fromkeys([k for k in agg_hyperlinks if len(k) > 2])
    for k in agg_hyperlinks:
        hlinks_superset[k] = []
        hlinks_ho_timestamps[k] = []

    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[i]]):  # record the timestamps of higher-order (>2) links
        for hlink in hlinks:
            if len(hlink) == 2:
                continue
            hlinks_ho_timestamps[hlink].append(t)
    hlinks_ho_weight = {k: len(v) for k, v in hlinks_ho_timestamps.items()}

    for hlink in agg_hyperlinks:  # construct a superset and subset.
        for hlink_ho in hlinks_ho_timestamps:
            if len(hlink) + 1 == len(hlink_ho):
                remaining_nodes = hlink_ho - hlink
                if len(remaining_nodes) == 1:
                    hlinks_superset[hlink].append(list(remaining_nodes)[0])

    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[i]]):  # for each pairwise link, calculate score along the tree
        for hlink in hlinks:
            if len(hlink) == 2:  # update increased score for hlink and its parental links
                metrics_now = dict().fromkeys([hlink], 1)
                hlink_stack = []  # stack used to traverse the tree
                prev_hlink_stack = []
                prev_t_stack = []
                for n1 in hlinks_superset[hlink]:
                    hl3 = hlink.union([n1])
                    hlink_stack.append(hl3)
                    prev_hlink_stack.append(hlink)
                    prev_t_stack.append(t)
                    metrics_now[hl3] = 0
                hlink_ho_tid = dict.fromkeys(hlink_stack, 0)
                while len(hlink_stack) > 0:
                    hlink_current = hlink_stack[-1]  # current traversed node in the tree
                    hlink_tid = hlink_ho_tid[hlink_current]  # at which timestamp
                    if hlink_tid == hlinks_ho_weight[hlink_current]:  # all timestamps have been done
                        hlink_stack.pop()
                        prev_hlink_stack.pop()
                        prev_t_stack.pop()
                    else:
                        tc = hlinks_ho_timestamps[hlink_current][hlink_tid]
                        if tc > prev_t_stack[-1]:  # update score for current node
                            metrics_now[hlink_current] += metrics_now[prev_hlink_stack[-1]]
                        hlink_ho_tid[hlink_current] += 1  # update tid
                        for n1 in hlinks_superset[hlink_current]:  # push next
                            hl3 = hlink_current.union([n1])
                            hlink_stack.append(hl3)
                            prev_hlink_stack.append(hlink_current)
                            prev_t_stack.append(tc)
                            hlink_ho_tid[hl3] = 0
                            if hl3 not in metrics_now:
                                metrics_now[hl3] = 0
                for k in metrics_now:
                    metrics[k] += metrics_now[k]

    return metrics


def two_hop_score_order2(h_tnet: hyperTN, i=0) -> dict:
    '''
    calculate the 2-walk metric for pairwise links, which is also #2-hop time-respecting walks (minimum hop paths)
    :param h_tnet:
    :param alpha:
    :param i:
    :return:
    '''
    with open(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 't_division.json'), 'r') as f:
        ts = json.loads(f.read().rstrip('\n'))
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
    hlinks_order2_weights_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)  # link weight at time t
    hlinks_order2_metric = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    adj_mat_order2 = np.zeros((h_tnet.n, h_tnet.n), dtype=bool)  # adjacency matrix encoding pairwise connection
    for hlinks in h_tnet.hypercontacts[:ts[i]]:
        for hlink in hlinks:
            if len(hlink) > 2:
                continue
            node1, node2 = hlink
            adj_mat_order2[node1, node2] = True
            adj_mat_order2[node2, node1] = True

    for hlinks in h_tnet.hypercontacts[:ts[i]]:
        for hlink in hlinks:  # update the link weights at each time
            if len(hlink) > 2:
                continue
            hlinks_order2_weights_t[hlink] += 1
        for hlink in hlinks:
            if len(hlink) > 2:
                continue
            node1, node2 = hlink
            for node in np.nonzero(adj_mat_order2[node1])[0]:
                if node == node2:
                    continue
                link_toupdate = frozenset([node, node1])
                hlinks_order2_metric[link_toupdate] += agg_hyperlinks[link_toupdate] - hlinks_order2_weights_t[
                    link_toupdate]
            for node in np.nonzero(adj_mat_order2[node2])[0]:
                if node == node1:
                    continue
                link_toupdate = frozenset([node, node2])
                hlinks_order2_metric[link_toupdate] += agg_hyperlinks[link_toupdate] - hlinks_order2_weights_t[
                    link_toupdate]

    return hlinks_order2_metric


if __name__ == '__main__':
    dataset = 'infectious'
    h_tnet = hyperTN(dataset)
    # metrics1 = effective_weights_order3(h_tnet)
    # metrics2 = local_effective_weights(h_tnet)