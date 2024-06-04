import os
import re

import numpy as np

from pathlib import Path


def get_cross_section_and_nevent(path):
    # return cross section and number in banner file
    # path: run path
    name = os.path.split(path)[1]

    with open(os.path.join(path, f'{name}_tag_1_banner.txt')) as f:
        for line in f.readlines():
                
            #  Integrated weight (pb)  :       0.020257
            match = re.match('#  Integrated weight \(pb\)  : +(\d+\.\d+)', line)
            if match:
                # unit: fb
                cross_section = float(match.group(1)) * 1000
            # #  Number of Events        :       100000
            match = re.match('#  Number of Events        :       (\d+)', line)
            if match:
                # unit: fb
                nevent = int(match.group(1))
    
    return cross_section, nevent


def compute_nevent_in_SR_SB(sensitivity=1.0, L=139):
    # return number of signal and background events in signal region and sideband region
    sample_dir = Path('../Sample/HVmodel/data/')

    results_b = np.load(sample_dir / 'selection_results_SB_4400_5800_b.npy', allow_pickle=True).item()
    results_s = np.load(sample_dir / 'selection_results_SB_4400_5800_s.npy', allow_pickle=True).item()

    # Total cross section and number of events
    xection, _ = get_cross_section_and_nevent('../Sample/ppjj/Events/run_03')

    # cross section in signal region and sideband region
    cross_section_SR = results_b['cutflow_number']['Signal region'] / results_b['cutflow_number']['Total'] * xection
    cross_section_SB = results_b['cutflow_number']['Sideband region'] / results_b['cutflow_number']['Total'] * xection
    print(f'Background cross section, SR: {cross_section_SR:.2f} fb, SB: {cross_section_SB:.2f} fb')

    # number of background events in signal region and sideband region
    n_SR_B = cross_section_SR * L
    n_SB_B = cross_section_SB * L

    print(f'Background sample size: SR: {n_SR_B:.1f}, SB: {n_SB_B:.1f}')

    n_SR_S = sensitivity * np.sqrt(n_SR_B)
    n_SB_S = n_SR_S * results_s['cutflow_number']['Sideband region'] / results_s['cutflow_number']['Signal region']
    print(f'Signal sample size: SR: {n_SR_S:.1f}, SB: {n_SB_S:.1f}')

    return n_SR_S, n_SR_B, n_SB_S, n_SB_B


def load_samples(path):
    root, _ = os.path.splitext(path)
    X = np.load(f'{root}-data.npy')
    Y = np.load(f'{root}-label.npy')
    return X, Y


def get_SR_SB_sample_from_npy(npy_dirs: list, nevents: tuple, seed=0):
    # npy_dirs: list of npy directories
    # nevents: tuple of (n_sig_SR, n_sig_SB, n_bkg_SR, n_bkg_SB)

    npy_dir0 = Path(npy_dirs[0])

    data_sig_SR = np.load(npy_dir0 / 'sig_in_SR-data.npy')
    data_sig_SB = np.load(npy_dir0 / 'sig_in_SB-data.npy')
    data_bkg_SR = np.load(npy_dir0 / 'bkg_in_SR-data.npy')
    data_bkg_SB = np.load(npy_dir0 / 'bkg_in_SB-data.npy')

    data_SR = np.array([]).reshape(0, *data_sig_SR.shape[1:])
    label_SR = np.array([]).reshape(0)
    data_SB = np.array([]).reshape(0, *data_sig_SB.shape[1:])
    label_SB = np.array([]).reshape(0)

    n_sig_SR, n_sig_SB, n_bkg_SR, n_bkg_SB = nevents

    np.random.seed(seed)
    idx_sig_SR = np.random.choice(data_sig_SR.shape[0], n_sig_SR, replace=False)
    idx_sig_SB = np.random.choice(data_sig_SB.shape[0], n_sig_SB, replace=False)
    idx_bkg_SR = np.random.choice(data_bkg_SR.shape[0], n_bkg_SR, replace=False)
    idx_bkg_SB = np.random.choice(data_bkg_SB.shape[0], n_bkg_SB, replace=False)

    print(f'Preparing dataset from {npy_dirs}')
    for npy_dir in npy_dirs:
        npy_dir = Path(npy_dir)

        data_sig_SR = np.load(npy_dir / 'sig_in_SR-data.npy')
        data_sig_SB = np.load(npy_dir / 'sig_in_SB-data.npy')
        data_bkg_SR = np.load(npy_dir / 'bkg_in_SR-data.npy')
        data_bkg_SB = np.load(npy_dir / 'bkg_in_SB-data.npy')

        new_data_SR = np.concatenate([data_sig_SR[idx_sig_SR], data_bkg_SR[idx_bkg_SR]], axis=0)
        data_SR = np.concatenate([data_SR, new_data_SR], axis=0)

        new_label_SR = np.zeros(n_sig_SR + n_bkg_SR)
        new_label_SR[:n_sig_SR] = 1
        label_SR = np.concatenate([label_SR, new_label_SR])

        new_data_SB = np.concatenate([data_sig_SB[idx_sig_SB], data_bkg_SB[idx_bkg_SB]], axis=0)
        data_SB = np.concatenate([data_SB, new_data_SB], axis=0)

        new_label_SB = np.zeros(n_sig_SB + n_bkg_SB)
        new_label_SB[:n_sig_SB] = 1
        label_SB = np.concatenate([label_SB, new_label_SB])

    return data_SR, label_SR, data_SB, label_SB


def get_fpr_thresholds(y_true, y_scores):
    # get the false positive rate and corresponding threshold values

    # transform the input to numpy array
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores).reshape(-1)

    # obtain the thresholds
    thresholds = np.sort(np.unique(y_scores))
    
    # get negatives index
    negatives = (y_true == 0)
    negatives_count = np.sum(negatives)
    
    # pass_matrix shape: (n_thresholds, n_samples)
    pass_matrix = y_scores >= thresholds[:, None]
    negative_pass = np.sum(pass_matrix & negatives, axis=1)
    
    fpr = negative_pass / negatives_count
    
    return fpr, thresholds


def get_threshold_from_fpr(fpr, th, passing_rate=0.01):
    # th 由小到大，fpr 由大到小
    passing_rate_idx = (fpr > passing_rate).sum()
    return th[passing_rate_idx]