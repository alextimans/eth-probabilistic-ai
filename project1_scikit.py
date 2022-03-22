import typing
import random
import numpy as np
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans

# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class Model(object):

    def __init__(self):
        self.rng = np.random.default_rng(seed=0)
        # Optimal kernel as per CV
        self.kernel = Matern(1.0, (1e-5, 1e4), nu=2.5) + WhiteKernel(1.0, (1e-5, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           normalize_y=True,
                                           n_restarts_optimizer=5,
                                           copy_X_train=False,
                                           random_state=22)

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gp_mean, gp_std = self.gp.predict(x, return_std=True)
        pred = predict_cost_align(gp_mean, gp_std)

        return pred, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        # Sample reduction via k-means
        train_x, train_y = samp_reduce(train_x, train_y)
        print('Sample size reduced: %s / %s' %(train_x.shape, train_y.shape))

        self.gp.fit(train_x, train_y)
        print('Kernel:', self.gp.kernel_)


def samp_reduce(train_x, train_y, k=2000):
    """
    Reduces the sample size via k-means clustering
    to deal with GP inference complexity.

    :k: desired sample size
    """

    cluster = KMeans(n_clusters=k)
    cluster.fit(train_x, train_y)

    train_y_new = np.empty((k, ))
    for i in range(0, k):
        idx = (cluster.labels_ == i)
        train_y_new[i] = train_y[idx].mean()
    train_x_new = cluster.cluster_centers_
    return train_x_new, train_y_new


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :returns: mean asymmetric MSE cost
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case 1: overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case 2: true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case 3: else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    return np.mean(cost * weights)


def predict_cost_align(gp_mean: np.ndarray, gp_std:np.ndarray) -> np.ndarray:
    """
    Custom prediction assignment based on asymmetric cost.
    If threshold value within one std of pred mean, then pred = threshold.
    Else attempt to underpredict by subtracting a fraction of the std.

    :returns: predictions based on above rule
    """
    interval = np.column_stack((gp_mean - 1*gp_std, gp_mean + 1*gp_std))
    thresh_mask = (interval[:,0] <= THRESHOLD) & (THRESHOLD <= interval[:,1])

    pred = np.zeros(len(gp_mean))
    pred[thresh_mask] = THRESHOLD
    pred[~thresh_mask] = gp_mean[~thresh_mask] - 0.75*gp_std[~thresh_mask]
    print('Prediction aligned for asymmetric cost')

    return pred


def cross_val():
    """
    Custom 3-fold CV that uses one fold for training
    and two folds for validation due to GP inference complexity.
    CV on a set of potential kernel choices.

    :returns: CV results
    """

    x_train_full = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    y_train_full = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)

    random.seed(22)
    idx, samp, nfolds = list(range(0, len(y_train_full))), int(len(idx)/3), 3
    fold1_idx = np.array(random.sample(idx, samp))
    fold2_idx = np.array(random.sample(list(np.delete(idx, fold1_idx)), samp))
    fold3_idx = np.array(random.sample(list(np.delete(idx, np.concatenate([fold1_idx, fold2_idx]))), samp))
    folds = [fold1_idx, fold2_idx, fold3_idx]

    kernels = [
        RBF(1.0, (1e-5, 1e3)) + WhiteKernel(1.0, (1e-4, 1e2)),
        Matern(1.0, (1e-4, 1e4), nu=0.5) + WhiteKernel(1.0, (1e-4, 1e2)),
        Matern(1.0, (1e-4, 1e4), nu=1.5) + WhiteKernel(1.0, (1e-4, 1e2)),
        Matern(1.0, (1e-4, 1e4), nu=2.5) + WhiteKernel(1.0, (1e-4, 1e2)),
        RationalQuadratic(1.0, 1.0, (1e-5, 1e3), (1e-5, 1e3)) + WhiteKernel(1.0, (1e-4, 1e2)),
        #ExpSineSquared() + WhiteKernel(1.0, (1e-4, 1e2))
        ]
    
    cv_scores = np.empty((len(kernels), nfolds+1))
    
    for k in range(0, len(kernels)):
    
        print('Testing kernel:', kernels[k])
        gp = GaussianProcessRegressor(kernel=kernels[k],
                                      normalize_y=True,
                                      n_restarts_optimizer=5,
                                      copy_X_train=False,
                                      random_state=99)
        fold_scores = np.zeros(nfolds)
    
        for i in range(0, nfolds):
    
            x_train, y_train = x_train_full[folds[i], :], y_train_full[folds[i]]
            x_test = np.delete(x_train_full, folds[i], 0)
            y_test = np.delete(y_train_full, folds[i])
    
            print('Training...')
            gp.fit(x_train, y_train)
            print('Kernel optimized params:', gp.kernel_)
            gp_mean, gp_std = gp.predict(x_test, return_std=True)
            y_pred = predict_cost_align(gp_mean, gp_std)
    
            cost = cost_function(y_test, y_pred)
            fold_scores[i] = cost
            print('Cost of Fold %i : %f' %(i, cost))
    
        cv_scores[k] = np.append(fold_scores, np.mean(fold_scores))
    print('CV done for all kernels.\n', cv_scores)

    # CV results; rows = kernels, columns = 3 folds + mean
    # [[18.64331321 20.28387959 22.9342041  20.62046563]
    # [21.66303535 24.74084573 23.89273157 23.43220422]
    # [16.81005504 18.48160655 20.03845807 18.44337322]
    # [16.10545999 18.19062531 19.46788336 17.92132288]
    # [16.69982638 18.09239912 19.4924172  18.0948809 ]]


def main():
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)


if __name__ == "__main__":
    main()
