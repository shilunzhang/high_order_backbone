from collections import Counter
from random import sample
import numpy as np
from scipy.stats import rankdata

def overlap_top_N(seq1, seq2, top_n) -> float:  # TODO
    '''
    overlap of top_n elements in two sequences.
    :param seq1:
    :param seq2:
    :param top_n:
    :return:
    '''
    assert top_n < len(seq1) and top_n < len(seq2)
    seq1_ranking = 1+len(seq1)-rankdata(seq1, method='min')
    seq2_ranking = 1+len(seq2)-rankdata(seq2, method='min')
    # print(seq1_ranking, seq2_ranking)
    max_ranking1, max_ranking2 = len(seq1), len(seq2)
    for r in seq1_ranking:
        if r >= top_n and r < max_ranking1:
            max_ranking1 = r
    for r in seq2_ranking:
        if r >= top_n and r < max_ranking2:
            max_ranking2 = r
    assert max_ranking1 >= top_n and max_ranking2 >= top_n, print(max_ranking1, max_ranking2)
    # print(max_ranking1, max_ranking2)
    top_links1_a = np.where(seq1_ranking<max_ranking1)[0].tolist()
    top_links1_b = np.where(seq1_ranking==max_ranking1)[0].tolist()
    top_links2_a = np.where(seq2_ranking<max_ranking2)[0].tolist()
    top_links2_b = np.where(seq2_ranking==max_ranking2)[0].tolist()
    # print(top_links1_a, top_links1_b)
    # print(top_links2_a, top_links2_b)

    overlap = []
    for n in range(100):
        top_links1 = top_links1_a + sample(top_links1_b, k=top_n - len(top_links1_a))
        top_links2 = top_links2_a + sample(top_links2_b, k=top_n - len(top_links2_a))
        overlap.append(len(np.intersect1d(top_links1, top_links2)) / top_n)

    return np.mean(overlap)

def counter_total(ct: Counter) -> int:
    sum = 0
    for k in ct:
        sum += ct[k]

    return sum