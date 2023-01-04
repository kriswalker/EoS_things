import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize


def read_samples(path, params=None):
    datafile = open(path, 'r')
    data = []
    for i, line in enumerate(datafile):
        if i == 0:
            p = np.array([x for x in line.split()])
            if params is None:
                params = p
            pinds = np.array([np.argwhere((p == param)).flatten()[0]
                              for param in params])
        else:
            s = np.array([float(x) for x in line.split()])
            if s[np.argwhere((p == 'lambda_2'))] > s[
                    np.argwhere((p == 'lambda_1'))]:
                data.append(s[pinds])
    data = np.array(data)
    if np.shape(data)[1] == 1:
        data = data.flatten()
    return data, params


def kdeND(data, kernel, bandwidth, res):
    data_ = np.zeros(np.shape(data))

    def normalize(d):
        xmin = np.min(d)
        d_ = d - xmin
        xmax = np.max(d_)
        return d_ / xmax, xmin, xmax

    mins, maxs = [], []
    for i, di in enumerate(data.T):
        q = normalize(di)
        data_[:, i] = q[0]
        mins.append(q[1])
        maxs.append(q[2])

    # interpolation grid
    nvar = np.shape(data)[1]
    axes = [np.linspace(0, 1, res)] * nvar
    grid = np.meshgrid(*axes)
    coords = np.vstack(map(np.ravel, grid)).T
    if nvar > 2:
        coords[:, np.array([0, 1])] = coords[:, np.array([1, 0])]

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data_)
    log_density = kde.score_samples(coords).reshape(*[res]*nvar)
    log_density -= np.sum(np.log(maxs))

    axes_ = (axes * np.reshape(maxs, (nvar, 1))) + np.reshape(mins, (nvar, 1))

    return log_density, axes_


def invert(y, func, xguess):
    def min_func(x):
        return abs(y - func(x))
    sol = minimize(min_func, xguess)
    return sol.x[0]
