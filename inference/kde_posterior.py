import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import stats, signal, ndimage
from scipy.interpolate import interp2d


def read_samples(path):
    datafile = open(path, 'r')
    # results = result.read_in_result(filename=path + ,
    #                                 outdir=None, label=None,
    #                                 extension='json', gzip=False)
    data = []
    for i, line in enumerate(datafile):
        if i == 0:
            params = np.array([x for x in line.split()])
        else:
            s = np.array([float(x) for x in line.split()])
            if s[np.argwhere((params == 'lambda_2'))] > s[
                    np.argwhere((params == 'lambda_1'))]:
                data.append(s)
    return np.array(data), params


def interpolate(path, res, bw, plot=True, sup_data=None, event=''):

    data, params = read_samples(path)
    minds = np.argwhere((params == 'chirp_mass_source') |
                        (params == 'mass_ratio'))
    linds = np.argwhere((params == 'lambda_1') |
                        (params == 'lambda_2'))

    mdata = data[:, minds].flatten()
    mdata = mdata.reshape(int(len(mdata)/2), 2)
    ldata = data[:, linds].flatten()
    ldata = ldata.reshape(int(len(ldata)/2), 2)

    def kde_model(data, kernel, bandwidth, res=res, labels=(None, None, None)):
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)

        x1 = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), res)
        x2 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), res)
        x = (x1, x2)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.stack((X1.flatten(), X2.flatten()), axis=-1)
        log_density = kde.score_samples(X)
        density = np.exp(log_density).reshape(res, res)

        if plot:
            plt.pcolor(x1, x2, density)
            plt.scatter(data[:, 0], data[:, 1], marker='.', alpha=0.5, s=0.5)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.tight_layout()
            plt.savefig('kde_plots/{0}_kde_2d_{1}.png'.format(
                labels[2], event), dpi=200)
            plt.close()

            for i in range(2):
                plt.subplot(1, 2, i+1)
                density_i = np.sum(density, axis=i)
                # density_max = np.mean(np.sum(density, axis=i))
                # hist_max = np.mean(np.histogram(data[:,i], bins=50)[0])
                plt.hist(data[:, i], bins='auto', density=True)
                plt.plot(x[i], density_i / (np.sum(density_i) *
                                            (x[i][1] - x[i][0])))
                # plt.xlim(np.min(data), np.max(data))
                plt.xlabel(labels[i])
                plt.ylabel('N')
            plt.tight_layout()
            plt.savefig('kde_plots/{0}_kde_1d_{1}.png'.format(
                labels[2], event), dpi=200)
            plt.close()

        return kde, x1, x2, log_density.reshape(res, res).T

    def trapezoid(data, h, n, a, b, c, d, res=res, labels=(None, None, None)):
        x1 = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), res)
        x2 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), res)
        x = (x1, x2)

        x1_width = max(x1) - min(x1)
        x2_width = max(x2) - min(x2)
        grad = x2_width / x1_width

        angle = np.arctan(grad)
        length = x1_width / np.cos(np.arctan(grad))
        density = np.zeros((res, res))
        # dy = w / int(res/10)
        dy = 1 / n
        density[int(res/2)-int(n/2):int(res/2)+int(n/2), :] = h
        for i in range(int(a / dy)):
            density[int(res/2)+(int(n/2)+i), :] = h * (-(dy * i / a) + 1)
        for i in range(int(b / dy)):
            density[int(res/2)-(int(n/2)+i), :] = h * (-(dy * i / b) + 1)

        trap = np.zeros(res)
        trap[int(c / dy):-int(d / dy)] = 1
        for i in range(int(c / dy)):
            trap[int(c / dy)-i] = (-(dy * i / c) + 1)
        for i in range(int(d / dy)):
            trap[-int(d / dy)+i] = (-(dy * i / d) + 1)

        for i, row in enumerate(density):
            density[i] = row * trap

        # plt.pcolor(x1, x2, density)
        # plt.show()

        density = ndimage.rotate(density, angle * 180 / np.pi)
        # x_cen = int(np.shape(density)[0] / 2)
        # y_cen = int(np.shape(density)[1] / 2)
        # density = density[x_cen - int(res * np.sin(angle)/2):x_cen +int(res * np.sin(angle)/2),
        #                   y_cen - int(res * np.cos(angle)/2):y_cen + int(res * np.cos(angle)/2)]
        density = density[~np.all(density < 0.01, axis=1)]
        density = density[:, ~np.all(density < 0.01, axis=0)]
        crop = 0
        col = density.T[crop]
        while len(np.argwhere((col >= h))) == 0:
            crop += 1
            col = density.T[crop]
        crop += 4
        density = density[:, crop:] 
        # plt.pcolor(np.linspace(1, np.shape(density)[1], np.shape(density)[1]), 
        #            np.linspace(1, np.shape(density)[0], np.shape(density)[0]),
        #            density)
                   
        # plt.show()

        x_ = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]),
                         np.shape(density)[1])
        y_ = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]),
                         np.shape(density)[0])
        interp = interp2d(x_, y_, density)
        density = interp(x1, x2)
        if plot:
            plt.pcolor(x1, x2, density)
            plt.scatter(data[:, 0], data[:, 1], marker='.', alpha=0.5, s=0.5)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.tight_layout()
            plt.savefig('kde_plots/{0}_kde_2d_{1}.png'.format(
                labels[2], event), dpi=200)
            plt.close()

            for i in range(2):
                plt.subplot(1, 2, i+1)
                density_i = np.sum(density, axis=i)
                plt.hist(data[:, i], bins='auto', density=True)
                plt.plot(x[i], density_i / (np.sum(density_i) *
                                            (x[i][1] - x[i][0])))
                # plt.xlim(np.min(data), np.max(data))
                plt.xlabel(labels[i])
                plt.ylabel('N')
            plt.tight_layout()
            plt.savefig('kde_plots/{0}_kde_1d_{1}.png'.format(
                labels[2], event), dpi=200)
            plt.close()

        return x1, x2, density

    mu = np.min(mdata[:, 0])
    mdata[:, 0] = mdata[:, 0] - mu
    fac = (np.max(mdata[:, 1]) - np.min(mdata[:, 1])) / \
        (np.max(mdata[:, 0]) - np.min(mdata[:, 0]))
    mdata[:, 0] = mdata[:, 0] * fac

    print('Performing KDE...')
    bwm = bw[0] * 1.06 * np.std(mdata[:, 0]) * np.shape(mdata)[0]**(-1/5)
    # ‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’
    mkde, q, cm, mdens = kde_model(mdata, 'gaussian', bwm, res=res,
                                   labels=('$\\mathcal{M}$', '$q$',
                                           'chirpmass'))
    q = (q / fac) + mu
    mdens += np.log(fac)

    width = ((np.max(ldata[:, 0]) - np.min(ldata[:, 0])) +
             (np.max(ldata[:, 1]) - np.min(ldata[:, 1]))) / 2
    bwl = bw[1] * width
    lkde, l1, l2, ldens = kde_model(ldata, 'linear', bwl, res=res,
                                    labels=('$\\Lambda_1$', '$\\Lambda_2$',
                                            'lambda'))

    # height = sup_data[0]
    # n = sup_data[1]
    # a = sup_data[2]
    # b = sup_data[3]
    # c = sup_data[4]
    # d = sup_data[5]
    # l1, l2, ldens = trapezoid(ldata, height, n, a, b, c, d,
    #                           labels=('$\\Lambda_1$', '$\\Lambda_2$'))

    # density = np.einsum('ba,dc->abcd', mdens, ldens)

    # dvol = (m1[1] - m1[0]) * (m2[1] - m2[0]) * (l1[1] - l1[0]) * (l2[1] - l2[0])
    # integral = np.sum(density.flatten() * dvol)
    # density /= integral
    # print('integral =', integral)

    return (mdens, ldens), (cm, q), (l1, l2)
