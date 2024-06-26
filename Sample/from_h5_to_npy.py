#!/usr/bin/env python
# coding: utf-8
# python from_h5_to_npy.py <h5_path> <output_path> <resolution>
# python from_h5_to_npy.py ./HVmodel/data/DA/mix_sample_0.0_aug_3.h5 ./HVmodel/data/DA/mix_sample_0.0_aug_3.npy 75

import os
import sys
import h5py
import random

import numpy as np


def std_phi(phi):
    # return the phi in range [-pi, pi]
    while np.any(phi > np.pi):
        phi[phi > np.pi] -= 2 * np.pi
    while np.any(phi < -np.pi):
        phi[phi < -np.pi] += 2 * np.pi
    return phi


def quadrant_max_vectorized(eta, phi, pt):
    # 建立條件列表
    # eta, phi shape: (n_events, n_constituents)
    # pt_quadrants shape: (n_events, 4)
    conditions = [
        (eta > 0) & (phi > 0),
        (eta > 0) & (phi < 0),
        (eta < 0) & (phi < 0),
        (eta < 0) & (phi > 0)
    ]

    # 建立輸出陣列
    pt_quadrants = np.zeros((eta.shape[0], 4))

    # 對每個象限進行操作
    for i, condition in enumerate(conditions):
        pt_quadrants[:, i] = np.sum(np.where(condition, pt, 0), axis=1)

    return pt_quadrants


def preprocess(pts, etas, phis):

    variances = np.var(phis, axis=1)
    phis = np.where((variances > 0.5)[:, None], phis + np.pi, phis)
    phis = std_phi(phis)

    # compute pt weighted center
    # eta_central shape: (n_events, 1)
    eta_central = ((pts * etas).sum(axis=1) / pts.sum(axis=1))[:, None]
    phi_central = ((pts * phis).sum(axis=1) / pts.sum(axis=1))[:, None]

    # compute rotation angle
    s_etaeta = (pts * (etas - eta_central)**2).sum(axis=1) / pts.sum(axis=1)
    s_phiphi = (pts * (phis - phi_central)**2).sum(axis=1) / pts.sum(axis=1)
    s_etaphi = (pts * (etas - eta_central) * (phis - phi_central)).sum(axis=1) / pts.sum(axis=1)

    angle = -np.arctan2(-s_etaeta + s_phiphi + np.sqrt((s_etaeta - s_phiphi)**2 + 4. * s_etaphi**2), 2.*s_etaphi)[:, None]

    eta_shift, phi_shift = etas - eta_central, std_phi(phis - phi_central)
    eta_rotat, phi_rotat = eta_shift * np.cos(angle) - phi_shift * np.sin(angle), phi_shift * np.cos(angle) + eta_shift * np.sin(angle)

    pt_quadrants = quadrant_max_vectorized(eta_rotat, phi_rotat, pts)

    phi_flip = np.where((np.argmax(pt_quadrants, axis=1) == 1) | (np.argmax(pt_quadrants, axis=1) == 2), -1., 1.)[:, None]
    eta_flip = np.where((np.argmax(pt_quadrants, axis=1) == 2) | (np.argmax(pt_quadrants, axis=1) == 3), -1., 1.)[:, None]

    eta_news = eta_rotat * eta_flip
    phi_news = phi_rotat * phi_flip

    return pts, eta_news, phi_news


def pixelization(pts, etas, phis, res=75):
    # pixelate jet constituents
    # res: resolution of the image

    nevent = pts.shape[0]

    # 計算 bin 的邊界
    bins_eta = np.linspace(-1.0, 1.0, res + 1)
    bins_phi = np.linspace(-1.0, 1.0, res + 1)

    # 計算每個數據點在直方圖中的位置
    # shape: (nevent, MAX_JETS)
    bin_idx_eta = np.digitize(etas, bins_eta) - 1
    bin_idx_phi = np.digitize(phis, bins_phi) - 1

    # 計算每個 bin 的權重總和
    hpT = np.zeros((nevent, res + 1, res + 1))
    np.add.at(hpT, (np.arange(nevent)[:, None], bin_idx_eta, bin_idx_phi), pts)

    hpT = hpT[:, :res, :res]

    return hpT


def from_h5_to_npy(h5_path, output_path, res=75):
    # Generate the jet image from h5 file and save it to npy file
    # res: resolution of the jet image
    with h5py.File(h5_path, 'r') as f:

        print('Preprocessing J1')
        _, eta1, phi1 = preprocess(f['J1/pt'][:], f['J1/eta'][:], f['J1/phi'][:])
        print('Preprocessing J2')
        _, eta2, phi2 = preprocess(f['J2/pt'][:], f['J2/eta'][:], f['J2/phi'][:])

        print('Computing the histogram')
        hpT0 = pixelization(f['J1/pt'][:], eta1, phi1, res)
        hpT1 = pixelization(f['J2/pt'][:], eta2, phi2, res)

        # 將結果堆疊起來
        # data shpae: (nevent, res, res, 2)
        # label shape: (nevent,)
        data = np.stack([hpT0, hpT1], axis=-1)
        label = f['EVENT/signal'][:]

    # shuffle
    ind_list = list(range(len(label)))
    random.shuffle(ind_list)

    data = data[ind_list]
    label = label[ind_list]

    root, _ = os.path.splitext(output_path)

    print(f'Saving data to {root}-data.npy')
    np.save(f'{root}-data.npy', data)
    np.save(f'{root}-label.npy', label)


def main():

    h5_path = sys.argv[1]
    output_path = sys.argv[2]
    res = int(sys.argv[3])

    from_h5_to_npy(h5_path, output_path, res)


if __name__ == '__main__':
    main()
