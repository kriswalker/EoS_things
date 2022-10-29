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

    return x.flatten(), np.exp(log_density)
