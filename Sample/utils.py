import numpy as np


def std_phi(phi):
    # return the phi in range [-pi, pi]
    return np.mod(phi + np.pi, 2 * np.pi) - np.pi


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


def rotation(etas, phis, angles):
    # etas, phis are the coordinates
    # angles: rotation angle
    eta_rotat, phi_rotat = etas * np.cos(angles) - phis * np.sin(angles), phis * np.cos(angles) + etas * np.sin(angles)
    return eta_rotat, phi_rotat


def preprocess(pts, etas, phis):

    variances = np.var(phis, axis=1)
    phis = np.where((variances > 0.5)[:, None], phis + np.pi, phis)
    phis = std_phi(phis)

    # compute pt weighted center
    # eta_central shape: (n_events, 1)
    pt_sum = pts.sum(axis=1)
    eta_central = ((pts * etas).sum(axis=1) / pt_sum)[:, None]
    phi_central = ((pts * phis).sum(axis=1) / pt_sum)[:, None]

    # compute rotation angle
    s_etaeta = (pts * (etas - eta_central)**2).sum(axis=1) / pt_sum
    s_phiphi = (pts * (phis - phi_central)**2).sum(axis=1) / pt_sum
    s_etaphi = (pts * (etas - eta_central) * (phis - phi_central)).sum(axis=1) / pt_sum

    angle = -np.arctan2(-s_etaeta + s_phiphi + np.sqrt((s_etaeta - s_phiphi)**2 + 4. * s_etaphi**2), 2.*s_etaphi)[:, None]

    eta_shift, phi_shift = etas - eta_central, std_phi(phis - phi_central)
    eta_rotat, phi_rotat = rotation(eta_shift, phi_shift, angle)

    pt_quadrants = quadrant_max_vectorized(eta_rotat, phi_rotat, pts)

    phi_flip = np.where((np.argmax(pt_quadrants, axis=1) == 1) | (np.argmax(pt_quadrants, axis=1) == 2), -1., 1.)[:, None]
    eta_flip = np.where((np.argmax(pt_quadrants, axis=1) == 2) | (np.argmax(pt_quadrants, axis=1) == 3), -1., 1.)[:, None]

    eta_news = eta_rotat * eta_flip
    phi_news = phi_rotat * phi_flip

    return pts, eta_news, phi_news