import sys
import numpy as np
from bilby import result
from tqdm import tqdm
from scipy.interpolate import interp1d
from toast.piecewise_polytrope import density_of_pressure


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
