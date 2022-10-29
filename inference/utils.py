import numpy as np
from sklearn.neighbors import KernelDensity


def read_samples(path, params):
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

    return np.array(data), params


def generate_kde(data, kernel, bandwidth, res):
    xdata = data[:, 0]
    ydata = data[:, 1]

    # normalize data to range [0,1]
    xmin = np.min(xdata)
    xdata_ = xdata - xmin
    xmax = np.max(xdata_)
    data[:, 0] = xdata_ / xmax

    ymin = np.min(ydata)
    ydata_ = ydata - ymin
    ymax = np.max(ydata_)
    data[:, 1] = ydata_ / ymax

    # interpolation grid
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y)
    XY = np.stack((X.flatten(), Y.flatten()), axis=-1)

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
    log_density = kde.score_samples(XY).reshape(res, res)

    return (x * xmax) + xmin, (y * ymax) + ymin, log_density
