from os import path
from copy import copy
import numpy as np
from itertools import combinations
import argparse
from hyperTNet import *


def time_decayed_link_weight(h_tnet: HyperTN, alpha):
    ts = h_tnet.time_division(which='all')
    its = len(ts) - 1

    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    hlinks_metric = dict().fromkeys(agg_hyperlinks, 0)

    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[its]]):
        for hlink in hlinks:
            hlinks_metric[hlink] += (t+1)**(-alpha)

    return hlinks_metric

def effective_weights_order3(h_tnet: HyperTN, inverse=False, i=0) -> dict:
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


def local_ascending_path_metric(h_tnet: HyperTN, i=0) -> dict:
    '''
    calculate the ascending-order path metric for arbitrary order hyperlinks using a tree structure
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

def local_subplink_prod_metric(h_tnet: HyperTN, subnet=0) -> dict:
    '''
    Calculate the metric of the higher-order link based on its weight w_j and the weights of its sub-links as w_j*(w1+w2+w3)
    :param h_tnet: temporal hypergraph
    :param i: the sub-net index
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    for hlink in metrics:  # update higher-order links
        for hl in combinations(hlink, 2):
            hl = frozenset(hl)
            if hl in agg_hyperlinks:
                metrics[hlink] += agg_hyperlinks[hlink] * agg_hyperlinks[hl]
        metrics[hlink] += agg_hyperlinks[hlink]

    return metrics

def local_subplink_division_metric(h_tnet: HyperTN, subnet=0) -> dict:
    '''
    Calculate the metric of the higher-order link based on its weight w_j and the weights of its sub-links as w_j*(w1+w2+w3)
    :param h_tnet: temporal hypergraph
    :param i: the sub-net index
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    for hlink in metrics:  # update higher-order links
        sublink_weight = 0
        for hl in combinations(hlink, 2):
            hl = frozenset(hl)
            if hl in agg_hyperlinks:
                sublink_weight += agg_hyperlinks[hl]
        metrics[hlink] += agg_hyperlinks[hlink] / (1+sublink_weight)

    return metrics

def local_adjplink_prod_metric(h_tnet: HyperTN, subnet=0) -> dict:
    '''
    Calculate the metric of the 3rd order link based on its weight w_j and the weights of its adjacent-pairwise links as w_j*(w1+w2+w3),
    adjacent-pairwise links have at least on common node shared with the target link.
    :param h_tnet: temporal hypergraph
    :param i: the sub-net index
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    for hlink in metrics:  # update higher-order links
        plink_weight = 0
        for hl in agg_hyperlinks:
            if len(hl) == 2 and len(hl.intersection(hlink)) >= 1:  # filter out adjacent pairwise links
                plink_weight += agg_hyperlinks[hl]
        metrics[hlink] = agg_hyperlinks[hlink] * (1 + plink_weight)

    return metrics

def local_adjplink_division_metric(h_tnet: HyperTN, subnet=0) -> dict:
    '''
    Calculate the metric of the 3rd order link based on its weight w_j and the weights of its adjacent-pairwise links as w_j*(w1+w2+w3),
    adjacent-pairwise links have at least on common node shared with the target link.
    :param h_tnet: temporal hypergraph
    :param i: the sub-net index
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    for hlink in metrics:  # update higher-order links
        plink_weight = 0
        for hl in agg_hyperlinks:
            if len(hl) == 2 and len(hl.intersection(hlink)) >= 1:  # filter out adjacent pairwise links
                plink_weight += agg_hyperlinks[hl]
        metrics[hlink] = agg_hyperlinks[hlink] / (1+plink_weight)

    return metrics

def local_subsuplink_division_metric(h_tnet: HyperTN, subnet=0) -> dict:
    '''
    Calculate the metric of the 3rd order link based on its weight w_j and the weights of its adjacent-pairwise links as w_j*(w1+w2+w3),
    adjacent-pairwise links have at least on common node shared with the target link.
    :param h_tnet: temporal hypergraph
    :param i: the sub-net index
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    for hlink in metrics:  # update higher-order links
        subsuplink_weight = 0
        for hl in agg_hyperlinks:
            if hl.issubset(hlink) or hl.issuperset(hlink):
                subsuplink_weight += agg_hyperlinks[hl]
        metrics[hlink] = agg_hyperlinks[hlink] / (1+subsuplink_weight)

    return metrics

def local_adjlink_division_metric(h_tnet: HyperTN, subnet=0) -> dict:
    '''
    Calculate the metric of the 3rd order link based on its weight w_j and the weights of its adjacent links as w_j*(w1+w2+w3),
    adjacent links have at least on common node shared with the target link.
    :param h_tnet: temporal hypergraph
    :param i: the sub-net index
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    for hlink in metrics:  # update higher-order links
        adjlink_weight = 0
        for hl in agg_hyperlinks:
            if len(hl.intersection(hlink)) >= 1:
                adjlink_weight += agg_hyperlinks[hl]
        metrics[hlink] = agg_hyperlinks[hlink] / (1+adjlink_weight)

    return metrics

def local_subplink_prod_temporal_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective metric (sum of weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks], 0)

    hlinks_order2_superset = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2])  # for each link, record hyperlinks containing the link itself.
    for k in hlinks_order2_superset:
        hlinks_order2_superset[k] = []

    for hlink in hlinks_order2_superset:  # construct the superset of pairwise links.
        for hlink_ho in agg_hyperlinks:
            if len(hlink_ho) > 2 and hlink_ho.issuperset(hlink):
                hlinks_order2_superset[hlink].append(hlink_ho)

    hlink_weight_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) > 2:
                hlink_weight_t[hlink] += 1
        for hlink in hlinks:
            if len(hlink) == 2:
                for superset in hlinks_order2_superset[hlink]:
                    metrics[superset] += agg_hyperlinks[superset] - hlink_weight_t[superset]

    return metrics

def local_subplink_division_temporal_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective metric (sum of weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks], 0)

    subplinks_dict = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3])
    for k in subplinks_dict:
        subplinks_dict[k] = []

    for hlink in subplinks_dict:  # construct the superset of pairwise links.
        for hl in agg_hyperlinks:
            if len(hl)==2 and hl.issubset(hlink):
                subplinks_dict[hlink].append(hl)

    plink_weight_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) >= 3:
                subplink_weight = 0
                for hl in subplinks_dict[hlink]:
                    subplink_weight += plink_weight_t[hl]
                metrics[hlink] += 1/(1+subplink_weight)
        for hlink in hlinks:
            if len(hlink) == 2:
                plink_weight_t[hlink] += 1

    return metrics

def local_adjplink_prod_temporal_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective square sum metric (sum of squre weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    adjplinks_dict = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3])
    for k in adjplinks_dict:
        adjplinks_dict[k] = []

    for hlink in adjplinks_dict:  # construct the superset of pairwise links.
        for hl in agg_hyperlinks:
            if len(hl) == 2 and len(hl.intersection(hlink)) >= 1:
                adjplinks_dict[hlink].append(hl)

    adjplinks_weight_order2_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) >= 3:
                for hl in adjplinks_dict[hlink]:
                    metrics[hlink] += adjplinks_weight_order2_t[hl]

        for hlink in hlinks:
            if len(hlink) == 2:
                adjplinks_weight_order2_t[hlink] += 1

    return metrics

def local_adjplink_division_temporal_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective square sum metric (sum of squre weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    adjplinks_dict = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3])
    for k in adjplinks_dict:
        adjplinks_dict[k] = []

    for hlink in adjplinks_dict:  # construct the superset of pairwise links.
        for hl in agg_hyperlinks:
            if len(hl) == 2 and len(hl.intersection(hlink)) >= 1:
                adjplinks_dict[hlink].append(hl)

    adjplinks_weight_order2_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) >= 3:
                adjplink_weight = 0
                for hl in adjplinks_dict[hlink]:
                    adjplink_weight += adjplinks_weight_order2_t[hl]
                metrics[hlink] += 1/(1+adjplink_weight)

        for hlink in hlinks:
            if len(hlink) == 2:
                adjplinks_weight_order2_t[hlink] += 1

    return metrics

def local_adjlink_division_temporal_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective square sum metric (sum of squre weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    adjlinks_dict = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3])
    for k in adjlinks_dict:
        adjlinks_dict[k] = []

    for hlink in adjlinks_dict:  # construct the superset of pairwise links.
        for hl in agg_hyperlinks:
            if len(hl.intersection(hlink)) >= 1:
                adjlinks_dict[hlink].append(hl)

    linkweight_t = dict().fromkeys([k for k in agg_hyperlinks], 0.0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) >= 3:
                adjlink_weight = 0
                for hl in adjlinks_dict[hlink]:
                    adjlink_weight += linkweight_t[hl]
                metrics[hlink] += 1/(1+adjlink_weight)

        for hlink in hlinks:
            linkweight_t[hlink] += 1

    return metrics

def local_subsuplink_division_temporal_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective square sum metric (sum of squre weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3], 0.0)

    subsuplinks_dict = dict().fromkeys([k for k in agg_hyperlinks if len(k) >= 3])
    for k in subsuplinks_dict:
        subsuplinks_dict[k] = []

    for hlink in subsuplinks_dict:  # construct the superset of pairwise links.
        for hl in agg_hyperlinks:
            if hl.issubset(hlink) or hl.issuperset(hlink):
                subsuplinks_dict[hlink].append(hl)

    linkweight_t = dict().fromkeys([k for k in agg_hyperlinks], 0.0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) >= 3:
                subsuplink_weight = 0
                for hl in subsuplinks_dict[hlink]:
                    subsuplink_weight += linkweight_t[hl]
                metrics[hlink] += 1/(1+subsuplink_weight)

        for hlink in hlinks:
            linkweight_t[hlink] += 1

    return metrics

def local_subplink_square_prod_temporal_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective square sum metric (sum of squre weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks], 0)

    hlink_weight_order2_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) > 2:
                for subset in combinations(hlink, 2):
                    subset = frozenset(subset)
                    if subset in hlink_weight_order2_t:
                        metrics[hlink] += hlink_weight_order2_t[subset] ** 2

        for hlink in hlinks:
            if len(hlink) == 2:
                hlink_weight_order2_t[hlink] += 1

    return metrics

def local_effective_sum_inverse_metric(h_tnet: HyperTN, subnet) -> dict:
    ''' return the local effective metric (inverse of sum of weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks], 0)

    hlink_order2_weight_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) > 2:
                link_weights_sum = 0
                for link_tuple in combinations(hlink, 2):
                    link_pairwise = frozenset(link_tuple)
                    if link_pairwise in hlink_order2_weight_t:
                        link_weights_sum += hlink_order2_weight_t[link_pairwise]
                # if link_weights_sum > 0:
                metrics[hlink] += 1/(link_weights_sum+1)

        for hlink in hlinks:
            if len(hlink) == 2:
                hlink_order2_weight_t[hlink] += 1

    return metrics

def local_effective_sum_inverse_normalized_metric(h_tnet: HyperTN, subnet, coef=1) -> dict:
    ''' return the local effective metric (normalied by inverse of sum of weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks], 0)

    hlink_order2_weight_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) > 2:
                link_weights_sum = 0
                for link_tuple in combinations(hlink, 2):
                    link_pairwise = frozenset(link_tuple)
                    if link_pairwise in hlink_order2_weight_t:
                        link_weights_sum += hlink_order2_weight_t[link_pairwise]
                # if link_weights_sum > 0:
                metrics[hlink] += link_weights_sum / (coef * link_weights_sum + 1)

        for hlink in hlinks:
            if len(hlink) == 2:
                hlink_order2_weight_t[hlink] += 1

    return metrics

def local_effective_sum_exponent_normalized_metric(h_tnet: HyperTN, subnet, coef=1) -> dict:
    ''' return the local effective metric (normalied by inverse of sum of weights of pairwise links) of all higher-order links
    :param h_tnet:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    metrics = dict().fromkeys([k for k in agg_hyperlinks], 0)

    hlink_order2_weight_t = dict().fromkeys([k for k in agg_hyperlinks if len(k) == 2], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:
            if len(hlink) > 2:
                link_weights_sum = 0
                for link_tuple in combinations(hlink, 2):
                    link_pairwise = frozenset(link_tuple)
                    if link_pairwise in hlink_order2_weight_t:
                        link_weights_sum += hlink_order2_weight_t[link_pairwise]
                # if link_weights_sum > 0:
                metrics[hlink] += link_weights_sum * np.exp(-coef*link_weights_sum)

        for hlink in hlinks:
            if len(hlink) == 2:
                hlink_order2_weight_t[hlink] += 1

    return metrics

def local_cross_order_weight(h_tnet: HyperTN, i, supsub='all') -> dict:
    ''''''
    assert supsub in ['all', 'subset', 'superset']
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
    metric = dict().fromkeys(agg_hyperlinks, 0)

    for hlink in metric:
        subsupset = []
        if supsub == 'all':
            subsupset = [k for k in agg_hyperlinks if
                         (k.issubset(hlink) or k.issuperset(hlink)) and len(k) != len(hlink)]
        elif supsub == 'subset':
            subsupset = [k for k in agg_hyperlinks if k.issubset(hlink) and len(k) != len(hlink)]
        elif supsub == 'superset':
            subsupset = [k for k in agg_hyperlinks if k.issuperset(hlink) and len(k) != len(hlink)]
        if len(subsupset) > 0:
            metric[hlink] = agg_hyperlinks[hlink] * np.sum([agg_hyperlinks[s] for s in subsupset])

    return metric


def local_cross_order_effective_weight(h_tnet: HyperTN, i, supsub='all') -> dict:
    '''
    Local cross-order effective weight with temporal information.
    :param h_tnet:
    :param i:
    :return:
    '''
    assert supsub in ['all', 'subset', 'superset']
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
    metric = dict().fromkeys(agg_hyperlinks, 0)

    hlink_subsupset = dict().fromkeys(agg_hyperlinks)

    for hlink in hlink_subsupset:  # construct the sub/superset.
        if hlink_subsupset[hlink] is None:
            hlink_subsupset[hlink] = []
        for hlink_ in agg_hyperlinks:
            if len(hlink_) != len(hlink):
                if supsub == 'all' and (hlink_.issuperset(hlink) or hlink_.issubset(hlink)):
                    hlink_subsupset[hlink].append(hlink_)
                elif supsub == 'subset' and hlink_.issubset(hlink):
                    hlink_subsupset[hlink].append(hlink_)
                elif supsub == 'superset' and hlink_.issuperset(hlink):
                    hlink_subsupset[hlink].append(hlink_)

    hlink_weight_t = dict().fromkeys([k for k in agg_hyperlinks], 0)

    for hlinks in h_tnet.hypercontacts[:ts[i]]:
        for hlink in hlinks:
            hlink_weight_t[hlink] += 1
        for hlink in hlinks:
            for subsupset in hlink_subsupset[hlink]:
                metric[subsupset] += agg_hyperlinks[subsupset] - hlink_weight_t[subsupset]

    return metric

def local_effective_superset_metric(h_tnet: HyperTN, i, order, beta) -> dict:
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[i]])
    metric = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order], 0)
    hlinks_weight_t = dict().fromkeys(metric, 0)

    for hlinks in h_tnet.hypercontacts[:ts[i]]:
        for hlink in hlinks:
            if hlink in metric:
                hlinks_weight_t[hlink] += 1
        for hlink in hlinks:
            for hl in metric:
                if hlink.issuperset(hl):
                    metric[hl] += (0 + beta * (len(hlink)-2)) * (agg_hyperlinks[hl] - hlinks_weight_t[hl])

    return metric

def two_hop_walk_score(h_tnet: HyperTN, subnet) -> dict:
    '''
    calculate the 2-walk metric for all links, which is also #2-hop time-respecting walks (minimum hop paths)
    :param h_tnet:
    :param alpha:
    :return:
    '''
    ts = h_tnet.time_division(which='all')
    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[subnet]])
    hlinks_weights_t = dict().fromkeys([k for k in agg_hyperlinks], 0)  # link weight at time t
    metric = dict().fromkeys([k for k in agg_hyperlinks], 0)

    for hlinks in h_tnet.hypercontacts[:ts[subnet]]:
        for hlink in hlinks:  # update the link weights at each time
            hlinks_weights_t[hlink] += 1
        for hlink in hlinks:
            if len(hlink) > 2:  # only consider pairwise link two-hop away
                continue
            for hl in metric:
                if len(hl.intersection(hlink)) == 1:
                    metric[hl] += agg_hyperlinks[hl] - hlinks_weights_t[hl]

    return metric


def time_independent_link_local_metric(h_tnet: HyperTN, order, neighborhood='subplink', alpha=1.0, subnet=-1) -> dict:
    '''
    Time-independent local metric for higher-order links in temporal higher-order networks.
    :param h_tnet: the dataset
    :param neighborhood: type of neighborhood pairwise links considered in the metric
    :param alpha: the exponent applied to the weights of neighborhood pairwise links
    :param subnet: which subnetwork to
    :return: dictionary of metrics for each higher-order link
    '''
    if order == 2:
        assert neighborhood in {'adjplink', 'adjlink'}
    elif order > 2:
        assert neighborhood in {'subplink', 'adjplink'}

    ts = h_tnet.time_division(which='all')
    T = ts[subnet] if isinstance(subnet, int) else h_tnet.T

    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:T])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order], 0.0)

    # construct target neighborhood links which considered in the metric
    L = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order])
    for hlink in L:
        L[hlink] = []

    if neighborhood == 'subplink':  # then the target order > 2
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) == 2 and plink.issubset(hlink):
                    L[hlink].append(plink)
    elif neighborhood == 'adjplink':
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) == 2 and plink != hlink and len(plink.intersection(hlink)) >= 1:
                    L[hlink].append(plink)
    elif neighborhood == 'adjlink':  # then the target order is 2
        for hlink in L:
            for link in agg_hyperlinks:
                if link != hlink and len(link.intersection(hlink)) >= 1:
                    L[hlink].append(link)

    for hlink in metrics:  # update higher-order links
        links_weight = 0
        for link in L[hlink]:
            links_weight += agg_hyperlinks[link]
        metrics[hlink] = agg_hyperlinks[hlink] * (1 + links_weight)**alpha

    return metrics

def time_dependent_link_local_metric(h_tnet: HyperTN, order, neighborhood='subplink', alpha=1.0, subnet=-1) -> dict:
    '''
    Time-dependent local metric for higher-order links in temporal higher-order networks.
    :param h_tnet: the dataset
    :param neighborhood: type of neighborhood pairwise links considered in the metric
    :param alpha: the exponent applied to the weights of neighborhood pairwise links
    :param subnet: which subnetwork to
    :return: dictionary of metrics for each higher-order link
    '''
    if order == 2:
        assert neighborhood in {'adjplink', 'adjlink'}
    elif order > 2:
        assert neighborhood in {'subplink', 'adjplink'}

    ts = h_tnet.time_division(which='all')
    T = ts[subnet if isinstance(subnet, int) else -1]

    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:T])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order], 0)

    # construct target neighborhood links in the metric, L is a dict that records the target neighborhood
    L = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order])
    for hlink in L:
        L[hlink] = []

    if neighborhood == 'subplink':  # then the target order is > 2, i.e., len(hlink) > 2
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) == 2 and plink.issubset(hlink):
                    L[hlink].append(plink)
    elif neighborhood == 'adjplink':
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) ==2 and plink != hlink and len(plink.intersection(hlink)) >= 1:
                    L[hlink].append(plink)
    elif neighborhood == 'adjlink':  # then the target order is 2
        for hlink in L:
            for link in agg_hyperlinks:
                if link != hlink and len(link.intersection(hlink)) >= 1:
                    L[hlink].append(link)

    link_weight_t = dict().fromkeys([k for k in agg_hyperlinks], 0)  # the temporal link weight

    for hlinks in h_tnet.hypercontacts[:T]:
        for hlink in hlinks:
            if len(hlink) == order:
                neigh_link_weight = 0
                for hl in L[hlink]:
                    neigh_link_weight += link_weight_t[hl]
                metrics[hlink] += (1+neigh_link_weight)**alpha
        for hlink in hlinks:
            link_weight_t[hlink] += 1

    return metrics

def time_rescaled_weight(h_tnet: HyperTN, order, phi, subnet) -> dict:
    contacts = h_tnet.hypercontacts[:h_tnet.time_division(which='all')[subnet if isinstance(subnet, int) else -1]]
    agg_net = h_tnet.aggregate_hyperTN(contacts)
    metrics = dict().fromkeys([k for k in agg_net if len(k) == order], 0)

    for t, hlinks in enumerate(contacts):
        for hlink in hlinks:
            if hlink in metrics:
                metrics[hlink] += (t+1)**(-phi)

    return metrics

def time_independent_link_local_metric_randomization(h_tnet: HyperTN, model, randID, order, neighborhood='subplink', alpha=1.0, subnet=2) -> dict:
    '''
    Time-independent local metric for higher-order links in temporal higher-order networks.
    :param h_tnet: the dataset
    :param neighborhood: type of neighborhood pairwise links considered in the metric
    :param alpha: the exponent applied to the weights of neighborhood pairwise links
    :param subnet: which subnetwork to
    :return: dictionary of metrics for each higher-order link
    '''
    if order == 2:
        assert neighborhood in {'adjplink', 'adjlink'}
    elif order > 2:
        assert neighborhood in {'subplink', 'adjplink'}

    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.load_randomization(model, randID, subnet))
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order], 0.0)

    # construct target neighborhood links which considered in the metric
    L = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order])
    for hlink in L:
        L[hlink] = []

    if neighborhood == 'subplink':  # then the target order > 2
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) == 2 and plink.issubset(hlink):
                    L[hlink].append(plink)
    elif neighborhood == 'adjplink':
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) == 2 and plink != hlink and len(plink.intersection(hlink)) >= 1:
                    L[hlink].append(plink)
    elif neighborhood == 'adjlink':  # then the target order is 2
        for hlink in L:
            for link in agg_hyperlinks:
                if link != hlink and len(link.intersection(hlink)) >= 1:
                    L[hlink].append(link)

    for hlink in metrics:  # update higher-order links
        links_weight = 0
        for link in L[hlink]:
            links_weight += agg_hyperlinks[link]
        metrics[hlink] = agg_hyperlinks[hlink] * (1 + links_weight)**alpha

    return metrics

def time_dependent_link_local_metric_randomization(h_tnet: HyperTN, model, randID, order, neighborhood='subplink', alpha=1.0, subnet=2) -> dict:
    '''
    Time-dependent local metric for higher-order links in temporal higher-order networks.
    :param h_tnet: the dataset
    :param neighborhood: type of neighborhood pairwise links considered in the metric
    :param alpha: the exponent applied to the weights of neighborhood pairwise links
    :param subnet: which subnetwork to
    :return: dictionary of metrics for each higher-order link
    '''
    if order == 2:
        assert neighborhood in {'adjplink', 'adjlink'}
    elif order > 2:
        assert neighborhood in {'subplink', 'adjplink'}

    ts = h_tnet.time_division(which='all')
    T = ts[subnet if isinstance(subnet, int) else -1]

    hypercontacts_randomized = h_tnet.load_randomization(model, randID, subnet)
    agg_hyperlinks = h_tnet.aggregate_hyperTN(hypercontacts_randomized)
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order], 0)

    # construct target neighborhood links in the metric, L is a dict that records the target neighborhood
    L = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order])
    for hlink in L:
        L[hlink] = []

    if neighborhood == 'subplink':  # then the target order is > 2, i.e., len(hlink) > 2
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) == 2 and plink.issubset(hlink):
                    L[hlink].append(plink)
    elif neighborhood == 'adjplink':
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) ==2 and plink != hlink and len(plink.intersection(hlink)) >= 1:
                    L[hlink].append(plink)
    elif neighborhood == 'adjlink':  # then the target order is 2
        for hlink in L:
            for link in agg_hyperlinks:
                if link != hlink and len(link.intersection(hlink)) >= 1:
                    L[hlink].append(link)

    link_weight_t = dict().fromkeys([k for k in agg_hyperlinks], 0)  # the temporal link weight

    for hlinks in hypercontacts_randomized:
        for hlink in hlinks:
            if len(hlink) == order:
                neigh_link_weight = 0
                for hl in L[hlink]:
                    neigh_link_weight += link_weight_t[hl]
                metrics[hlink] += (1+neigh_link_weight)**alpha
        for hlink in hlinks:
            link_weight_t[hlink] += 1

    return metrics

def time_rescaled_weight_randomization(h_tnet: HyperTN, model, randID, order, phi, subnet=2) -> dict:
    hypercontacts_randomized = h_tnet.load_randomization(model, randID, subnet)
    agg_net = h_tnet.aggregate_hyperTN(hypercontacts_randomized)
    metrics = dict().fromkeys([k for k in agg_net if len(k) == order], 0)

    for t, hlinks in enumerate(hypercontacts_randomized):
        for hlink in hlinks:
            if hlink in metrics:
                metrics[hlink] += (t+1)**(-phi)

    return metrics

def time_dependent_time_propensity_link_local_metric(h_tnet: HyperTN, order, neighborhood='subplink', phi=-1.0, alpha=1.0, subnet=-1) -> dict:
    '''
    Time-dependent local metric for higher-order links in temporal higher-order networks.
    :param h_tnet: the dataset
    :param neighborhood: type of neighborhood pairwise links considered in the metric
    :param alpha: the exponent applied to the weights of neighborhood pairwise links
    :param subnet: which subnetwork to
    :return: dictionary of metrics for each higher-order link
    '''
    if order == 2:
        assert neighborhood in {'adjplink'}
    elif order > 2:
        assert neighborhood in {'subplink', 'adjplink'}

    ts = h_tnet.time_division(which='all')
    T = ts[subnet if isinstance(subnet, int) else -1]

    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:T])
    metrics = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order], 0)

    # construct target neighborhood links in the metric, L is a dict that records the target neighborhood
    L = dict().fromkeys([k for k in agg_hyperlinks if len(k) == order])
    for hlink in L:
        L[hlink] = []

    if neighborhood == 'subplink':  # then the target order is > 2, i.e., len(hlink) > 2
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) == 2 and plink.issubset(hlink):
                    L[hlink].append(plink)
    else:
        for hlink in L:
            for plink in agg_hyperlinks:
                if len(plink) ==2 and plink != hlink and len(plink.intersection(hlink)) >= 1:
                    L[hlink].append(plink)

    link_weight_t = dict().fromkeys([k for k in agg_hyperlinks], 0)  # the temporal link weight

    for t, hlinks in enumerate(h_tnet.hypercontacts[:T]):
        for hlink in hlinks:
            if len(hlink) == order:
                neigh_link_weight = 0
                for hl in L[hlink]:
                    neigh_link_weight += link_weight_t[hl]
                metrics[hlink] += ((t+1)**(-phi))*((1+neigh_link_weight)**alpha)
        for hlink in hlinks:
            link_weight_t[hlink] += 1

    return metrics

def appearance_time(h_tnet: HyperTN, subnet, which='min'):
    ts = h_tnet.time_division(which='all')
    T = h_tnet.T
    if isinstance(subnet, int):
        T = ts[subnet]
    agg_hnet = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:T])

    appear_time_dict = dict.fromkeys(agg_hnet)
    for hl in appear_time_dict:
        appear_time_dict[hl] = []

    for t, contacts in enumerate(h_tnet.hypercontacts[:T]):
        for contact in contacts:
            hl = frozenset(contact)
            appear_time_dict[hl].append(t)

    for hl in appear_time_dict:
        if which == 'min':
            appear_time_dict[hl] = np.min(appear_time_dict[hl])
        elif which == 'mean':
            appear_time_dict[hl] = np.mean(appear_time_dict[hl])

    return appear_time_dict

def optimal_corr_backbone_time_scaled_weight_over_coefs(h_tnet, subnet, beta, theta, order, eval_metric):
    # mu_list = [-3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2,
    #             0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    mu_list = np.round(np.linspace(-10.0, 10.0, 101), decimals=2).tolist()
    agg_net = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:h_tnet.time_division(which='all')[subnet if isinstance(subnet, int) else -1]])

    target_hlinks = {k: agg_net[k] for k in agg_net if len(k) == order}
    backbone = h_tnet.return_backbone({'beta': beta, 'theta': theta}, subnet=subnet)
    corrs = -np.ones(len(mu_list), dtype=np.float64)
    for p, phi in enumerate(mu_list):
        metrics_dict = time_rescaled_weight(h_tnet, order=order, phi=phi, subnet=subnet)
        data = [(metrics_dict[k], backbone[k] if k in backbone else 0) for k in target_hlinks]
        corrs[p] = eval_metric([e[0] for e in data], [e[1] for e in data])[0]

    return mu_list[np.argmax(corrs)], np.max(corrs)

def optimal_corr_backbone_metric_over_coefs(h_tnet, subnet, beta, theta, order, temporal, neighborhood, eval_metric):
    alpha_list = [-3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2,
                0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    agg_net = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:h_tnet.time_division(which='all')[subnet if isinstance(subnet, int) else -1]])

    target_hlinks = {k: agg_net[k] for k in agg_net if len(k) == order}
    backbone = h_tnet.return_backbone({'beta': beta, 'theta': theta}, subnet=subnet)
    corrs = -np.ones(len(alpha_list), dtype=np.float64)
    for a, alpha in enumerate(alpha_list):
        metrics_dict = dict()
        if temporal:
            metrics_dict = time_dependent_link_local_metric(h_tnet, order=order, neighborhood=neighborhood, alpha=alpha, subnet=subnet)
        else:
            metrics_dict = time_independent_link_local_metric(h_tnet, order=order, neighborhood=neighborhood, alpha=alpha, subnet=subnet)
        data = [(metrics_dict[k], backbone[k] if k in backbone else 0) for k in target_hlinks]
        corrs[a] = eval_metric([e[0] for e in data], [e[1] for e in data])[0]

    return alpha_list[np.argmax(corrs)], np.max(corrs)

def optimal_corr_backbone_combined_metric_over_coefs(h_tnet, subnet, beta, theta, order, neighborhood, eval_metric):
    phi_list = [-3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2,
                0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    alpha_list = [-3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2,
                0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    agg_net = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:h_tnet.time_division(which='all')[subnet if isinstance(subnet, int) else -1]])

    target_hlinks = {k: agg_net[k] for k in agg_net if len(k) == order}
    backbone = h_tnet.return_backbone({'beta': beta, 'theta': theta}, subnet=subnet)
    # corrs = np.random.uniform(0, 1, (len(phi_list), len(alpha_list)))
    corrs = -np.ones((len(phi_list), len(alpha_list)), dtype=np.float64)
    for p, phi in enumerate(phi_list):
        for a, alpha in enumerate(alpha_list):
            metrics_dict = time_dependent_time_propensity_link_local_metric(h_tnet, order=order, neighborhood=neighborhood, phi=phi, alpha=alpha, subnet=subnet)
            data = [(metrics_dict[k], backbone[k] if k in backbone else 0) for k in target_hlinks]
            corrs[p][a] = eval_metric([e[0] for e in data], [e[1] for e in data])[0]

    maxid_phi, maxid_alpha = np.argmax(corrs)//len(alpha_list), np.argmax(corrs)%len(phi_list)
    return (phi_list[maxid_phi], alpha_list[maxid_alpha]), np.max(corrs)


def parallel_grid_search_optimal_corr(h_tnet, subnet, theta, order, neighborhood, eval_metric, which):
    from joblib import Parallel, delayed
    from scipy.stats import kendalltau, pearsonr
    beta_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    corr_func = kendalltau if eval_metric=='kendalltau' else pearsonr
    if which == 'combined_metric':
        result = Parallel(n_jobs=len(beta_list), backend='loky')(delayed(optimal_corr_backbone_combined_metric_over_coefs)(h_tnet, subnet, beta, theta, order, neighborhood, corr_func) for beta in beta_list)

        params_optimal_as_beta = [e[0] for e in result]
        corr_optimal_as_beta = [e[1] for e in result]
        np.save(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'figs_data', 'optimal_{0}_params_combined_metric_{1}_as_beta_theta{2}_order{3}_subnet{4}.npy'.format(eval_metric, neighborhood, theta, order, subnet)), params_optimal_as_beta)
        np.save(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'figs_data', 'optimal_{0}corr_combined_metric_{1}_as_beta_theta{2}_order{3}_subnet{4}.npy'.format(eval_metric, neighborhood, theta, order, subnet)), corr_optimal_as_beta)
    elif which == 'time_scaled_weight':
        result = Parallel(n_jobs=len(beta_list), backend='loky')(delayed(optimal_corr_backbone_time_scaled_weight_over_coefs)(h_tnet, subnet, beta, theta, order, corr_func) for beta in beta_list)

        params_optimal_as_beta = [e[0] for e in result]
        corr_optimal_as_beta = [e[1] for e in result]
        np.save(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'figs_data',
                          'optimal_{0}_params_time_scaled_weight_as_beta_theta{1}_order{2}_subnet{3}_.npy'.format(
                              eval_metric, theta, order, subnet)), params_optimal_as_beta)
        np.save(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'figs_data',
                          'optimal_{0}corr_time_scaled_weight_as_beta_theta{1}_order{2}_subnet{3}_.npy'.format(
                              eval_metric, theta, order, subnet)), corr_optimal_as_beta)
    else:
        result = Parallel(n_jobs=len(beta_list), backend='loky')(delayed(optimal_corr_backbone_metric_over_coefs)(h_tnet, subnet, beta, theta, order, 'time_dependent' in which, neighborhood, corr_func) for beta in beta_list)

        params_optimal_as_beta = [e[0] for e in result]
        corr_optimal_as_beta = [e[1] for e in result]
        np.save(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'figs_data', 'optimal_{0}_params_{1}_{2}_as_beta_theta{3}_order{4}_subnet{5}.npy'.format(eval_metric, which, neighborhood, theta, order, subnet)), params_optimal_as_beta)
        np.save(path.join(PATH_TO_RESULTS, h_tnet.dataname, 'threshold_model', 'figs_data', 'optimal_{0}corr_{1}_{2}_as_beta_theta{3}_order{4}_subnet{5}.npy'.format(eval_metric, which, neighborhood, theta, order, subnet)), corr_optimal_as_beta)

    return result