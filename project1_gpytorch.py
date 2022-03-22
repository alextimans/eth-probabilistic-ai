import typing
import numpy as np
import torch
import gpytorch
from tqdm import tqdm

# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0

# Train and test constants
ITER = 1000
TEST_FREQ = 50
TEST_SIZE = 0.1
GRID_SIZE = 250


class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Option 1: "Classical" GP kernel
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #    gpytorch.kernels.RBFKernel() + 
        #    gpytorch.kernels.MaternKernel()
        # )

        # Option 2: Stuctured kernel interpolation
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x) if not GRID_SIZE else GRID_SIZE
        print(f'Grid size chosen: {grid_size}')
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.MaternKernel(),
                grid_size=grid_size,
                num_dims=2
            )
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        dist = gpytorch.distributions.MultivariateNormal(mean, cov)

        return dist


class Model(object):

    def __init__(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(torch.tensor(x, dtype=torch.float32)))
            gp_mean = posterior.mean.numpy()
            gp_std = torch.sqrt(posterior.covariance_matrix.diag()).numpy()
            
        pred = predict_cost_align(gp_mean, gp_std)
    
        return pred, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)

        if not self.model:
            self.model = ExactGPModel(train_x, train_y, self.likelihood)
        
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        # Loss for GPs - marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train loop
        with tqdm(range(ITER), bar_format='{bar}{r_bar}') as tloop:
            for i in tloop:
                optimizer.zero_grad()
                out = self.model(train_x)
                loss = -mll(out, train_y)

                loss.backward()
                optimizer.step()
                tloop.set_postfix(loss=loss.item())


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
