from collections import Counter
from random import sample
import numpy as np
from scipy.stats import rankdata


def jaccard_similarity(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))
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
    for n in range(10):
        top_links1 = top_links1_a + sample(top_links1_b, k=top_n - len(top_links1_a))
        top_links2 = top_links2_a + sample(top_links2_b, k=top_n - len(top_links2_a))
        overlap.append(len(np.intersect1d(top_links1, top_links2)) / top_n)

    return np.mean(overlap)

def cosine_similarity(arr1: np.array, arr2: np.array):
    return [np.dot(arr1, arr2) / (np.linalg.norm(arr1, ord=2) * np.linalg.norm(arr2, ord=2))]

def counter_total(ct: Counter) -> int:
    sum = 0
    for k in ct:
        sum += ct[k]

    return sum

def rename_dir_names(cwd: str):
    import os
    os.chdir(cwd)
    for bt in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if os.path.exists('beta1_{0:.2f}-beta2_{0:.2f}-theta_1.0'.format(bt)):
            os.rename('beta1_{0:.2f}-beta2_{0:.2f}-theta_1.0'.format(bt), 'beta1_{0:.3f}-beta2_{0:.3f}-theta_1.0'.format(bt))
        if os.path.exists('beta1_{0:.2f}-beta2_{0:.2f}-theta_-1.0'.format(bt)):
            os.rename('beta1_{0:.2f}-beta2_{0:.2f}-theta_-1.0'.format(bt), 'beta1_{0:.3f}-beta2_{0:.3f}-theta_-1.0'.format(bt))

if __name__ == '__main__':
    dataname = 'malawi'
    work_path = '/tudelft.net/staff-bulk/ewi/insy/MMC/shilunzhang/high_order_backbone/results/{0}/threshold_model/'.format(dataname)
    rename_dir_names(work_path)