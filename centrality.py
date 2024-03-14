import os.path
from copy import copy
import numpy as np
from itertools import combinations
import argparse
from hyperTNet import *


def time_decayed_link_weight(h_tnet: hyperTN, alpha):
    ts = h_tnet.time_division(which='all')
    its = len(ts) - 1

    agg_hyperlinks = h_tnet.aggregate_hyperTN(h_tnet.hypercontacts[:ts[its]])
    hlinks_metric = dict().fromkeys(agg_hyperlinks, 0)

    for t, hlinks in enumerate(h_tnet.hypercontacts[:ts[its]]):
        for hlink in hlinks:
            hlinks_metric[hlink] += (t+1)**(-alpha)

    return hlinks_metric

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


def local_ascending_path_metric(h_tnet: hyperTN, i=0) -> dict:
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

def local_subplink_prod_metric(h_tnet: hyperTN, subnet=0) -> dict:
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

def local_subplink_division_metric(h_tnet: hyperTN, subnet=0) -> dict:
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

def local_adjplink_prod_metric(h_tnet: hyperTN, subnet=0) -> dict:
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

def local_adjplink_division_metric(h_tnet: hyperTN, subnet=0) -> dict:
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

def local_subsuplink_division_metric(h_tnet: hyperTN, subnet=0) -> dict:
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

def local_adjlink_division_metric(h_tnet: hyperTN, subnet=0) -> dict:
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

def local_subplink_prod_temporal_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_subplink_division_temporal_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_adjplink_prod_temporal_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_adjplink_division_temporal_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_adjlink_division_temporal_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_subsuplink_division_temporal_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_subplink_square_prod_temporal_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_effective_sum_inverse_metric(h_tnet: hyperTN, subnet) -> dict:
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

def local_effective_sum_inverse_normalized_metric(h_tnet: hyperTN, subnet, coef=1) -> dict:
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

def local_effective_sum_exponent_normalized_metric(h_tnet: hyperTN, subnet, coef=1) -> dict:
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

def local_cross_order_weight(h_tnet: hyperTN, i, supsub='all') -> dict:
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


def local_cross_order_effective_weight(h_tnet: hyperTN, i, supsub='all') -> dict:
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

def local_effective_superset_metric(h_tnet: hyperTN, i, order, beta) -> dict:
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

def two_hop_walk_score(h_tnet: hyperTN, subnet) -> dict:
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


def time_independent_link_local_metric(h_tnet: hyperTN, order, neighborhood='subplink', alpha=1.0, subnet=-1) -> dict:
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

def time_dependent_link_local_metric(h_tnet: hyperTN, order, neighborhood='subplink', alpha=1.0, subnet=-1) -> dict:
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
    T = ts[subnet] if isinstance(subnet, int) else h_tnet.T

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

def appearance_time(h_tnet: hyperTN, subnet, which='min'):
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

def save_centrality_metrics(h_tnet: hyperTN, which, order, neighborhood='subplink', alpha=1.0, subnet=-1):
    import pickle
    global metric

    results_dir = path.join(PATH_TO_RESULTS, h_tnet.dataname, 'centrality')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    suffix_subnet = ''
    if isinstance(subnet, int):
        suffix_subnet = '_0.{0}'.format(subnet + 1 if subnet >= 0 else len(h_tnet.time_division(which='all')))

    if which == 'time_independent_link_local_metric':
        metric = time_independent_link_local_metric(h_tnet, order=order, neighborhood=neighborhood, alpha=alpha, subnet=subnet)
        with open(path.join(results_dir, 'T{0}-order{1}-{2}.pkl'.format(suffix_subnet, order, which+'_'+neighborhood+'_alpha'+str(alpha))), 'wb') as f:
            pickle.dump(metric, f)
    elif which == 'time_dependent_link_local_metric':
        metric = time_dependent_link_local_metric(h_tnet, order=order, neighborhood=neighborhood, alpha=alpha, subnet=subnet)
        with open(path.join(results_dir, 'T{0}-order{1}-{2}.pkl'.format(suffix_subnet, order, which+'_'+neighborhood+'_alpha'+str(alpha))), 'wb') as f:
            pickle.dump(metric, f)

    return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link centrality measures')
    parser.add_argument('--dataset', type=str, default='infectious', help='dataset')
    # parser.add_argument('--beta', type=float, default=0.01, help='infectivity for pairwise interaction')
    # parser.add_argument('--theta', type=float, default=2, help='threshold')
    parser.add_argument('--metric', type=str, default='time_independent_link_local_metric', help='which metric')
    parser.add_argument('--order', type=int, default=3, help='which order')
    parser.add_argument('--neighborhood', type=str, default='subplink', help='which neighborhood')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
    args = parser.parse_args()
    h_tnet = hyperTN(args.dataset)
    save_centrality_metrics(h_tnet, which=args.metric, order=args.order, neighborhood=args.neighborhood, alpha=args.alpha, subnet='T')
