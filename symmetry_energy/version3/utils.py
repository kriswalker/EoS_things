import numpy as np
from bilby import result
from tqdm import tqdm
from sklearn.neighbors import KernelDensity


def calc_confidence_interval(results_file, x, func, ndraws=100):

    results = result.read_in_result(filename=results_file)
    maxl_params = dict(results.posterior.iloc[
        results.posterior.log_likelihood.idxmax()])
    maxl_params.pop('log_likelihood')
    maxl_params.pop('log_prior')

    y_maxl = func(x, **maxl_params)
    draws = []
    ys = []
    for _ in tqdm(range(ndraws)):
        draw = results.posterior.sample().to_dict('records')[0]
        draw.pop('log_likelihood')
        draw.pop('log_prior')
        draws.append(draw)
        ys.append(func(x, **draw))
    ys = np.array(ys)
    ymean = np.mean(ys, axis=0)
    ystd = np.std(ys, axis=0)

    ci = []
    for i, y in enumerate(ymean):
        ci.append([y-ystd[i], y+ystd[i]])

    return np.array(ci), (ys, ymean, ystd)


def calc_model(x, func, models):
    ys = []
    models_ = []
    for model in models:
        try:
            y = func(x, model)
            ys.append(y)
            models_.append(model)
            print(model, "done")
        except:
            print(model, "failed")
            continue
    ys_dict = {}
    for i, key in enumerate(models_):
        ys_dict[key] = ys[i]
    return ys_dict


def kde1D(data, kernel, bandwidth, res):

    xhrange = (max(data) - min(data)) / 2
    xmid = (max(data) + min(data)) / 2
    x = np.linspace(xmid - 1.2 * xhrange, xmid + 1.2 * xhrange, res)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
    log_density = kde.score_samples(x).flatten()

    return np.exp(log_density), x.flatten()


def kde2D(data, kernel, bandwidth, res):
    data = np.copy(data)
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
    log_density -= (np.log(xmax) + np.log(ymax))

    return log_density, np.array([(x * xmax) + xmin, (y * ymax) + ymin])


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
