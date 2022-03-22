import random
import os
import typing
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


class BO_algo(object):
    def __init__(self):
        self.previous_points = []
        self.objective_model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.5) * RBF(1.5),
            alpha=0.01,
            optimizer=None,
            normalize_y=True) # GP model for objective function
        self.constraint_model = GaussianProcessRegressor(
            kernel=ConstantKernel(3.5) * RBF(2.0),
            alpha=0.005,
            optimizer=None,
            normalize_y=True) # GP model for constraint function


    def next_recommendation(self) -> np.ndarray:
        if not self.previous_points: # 1st sample is random initialization
            next_in = np.array([[np.random.uniform(0, 6), np.random.uniform(0, 6)]])
        else:
            next_in = self.optimize_acquisition_function()

        return next_in


    def optimize_acquisition_function(self) -> np.ndarray:
        def objective(x: np.array):
            return - self.acquisition_function(x)
        
        f_values = []
        x_values = []
        # Restarts the optimization 20 times and picks best solution
        for _ in range(20):
            x0 = domain_x[0, 0] + (domain_x[0, 1] - domain_x[0, 0]) * np.random.rand(1)
            x1 = domain_x[1, 0] + (domain_x[1, 1] - domain_x[1, 0]) * np.random.rand(1)
            result = fmin_l_bfgs_b(objective, x0=np.array([x0, x1]), bounds=domain_x,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain_x[0]))
            f_values.append(result[1])

        ind = np.argmin(f_values)

        return np.atleast_2d(x_values[ind])


    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        con_mean, con_std = self.constraint_model.predict(x.reshape(1,-1), return_std=True)
        con_prob_x = norm.cdf(0, con_mean, con_std) # Proba of x satisfying constraint c^(x) <= 0
        
        f_min = np.min([triplet[2] for triplet in self.previous_points]) # Get current min fct value
        
        obj_mean, obj_std = self.objective_model.predict(x.reshape(1,-1), return_std=True)
        z = (f_min - obj_mean) / obj_std # Standardized
        ei_x = obj_std * (z * norm.cdf(z) + norm.pdf(z)) # Expected improvement for obs x
        af_x = ei_x * con_prob_x # Acquisition function

        return af_x


    def add_data_point(self, x: np.ndarray, z: float, c: float):
        assert x.shape == (1, 2)
        self.previous_points.append([float(x[:, 0]), float(x[:, 1]), float(z), float(c)])

        x_vals = np.array([triplet[0:2] for triplet in self.previous_points])
        obj_vals = np.array([triplet[2] for triplet in self.previous_points])
        con_vals = np.array([triplet[3] for triplet in self.previous_points])
        
        self.objective_model.fit(x_vals, obj_vals)
        self.constraint_model.fit(x_vals, con_vals)


    def get_solution(self) -> np.ndarray:
        x_vals = np.array([triplet[0:2] for triplet in self.previous_points])
        obj_vals = np.array([triplet[2] for triplet in self.previous_points])
        con_vals = np.array([triplet[3] for triplet in self.previous_points])
        
        sorted_inds = np.argsort(obj_vals)
        feasible = False
        argmin_ind = 0
        for ind in sorted_inds:
            if con_vals[ind] <= 0:
                argmin_ind = ind
                feasible = True
                break

        if VERBOSE:
            if not feasible:
                print('Feasibility notifier: No feasible solution found!')
            print('Minimiser triplet: ', (x_vals[argmin_ind], obj_vals[argmin_ind], con_vals[argmin_ind]))
            print('Number of iterations: ', len(self.previous_points))
        
        return x_vals[argmin_ind]


domain_x = np.array([[0, 6], [0, 6]])
EVALUATION_GRID_POINTS = 250
CONSTRAINT_OFFSET = - 0.8  # Offset you can change to make the constraint more or less difficult
LAMBDA = 0.0
VERBOSE = False


def check_in_domain(x) -> bool:
    """Validate input"""
    x = np.atleast_2d(x)
    v_dim_0 = np.all(x[:, 0] >= domain_x[0, 0]) and np.all(x[:, 0] <= domain_x[0, 1])
    v_dim_1 = np.all(x[:, 1] >= domain_x[1, 0]) and np.all(x[:, 0] <= domain_x[1, 1])

    return v_dim_0 and v_dim_1


def f(x) -> np.ndarray:
    """Dummy objective"""
    l1 = lambda x0, x1: np.sin(x0) + x1 - 1

    return l1(x[:, 0], x[:, 1])


def c(x) -> np.ndarray:
    """Dummy constraint"""
    c1 = lambda x, y: np.cos(x) * np.cos(y) - 0.1

    return c1(x[:, 0], x[:, 1]) - CONSTRAINT_OFFSET


def get_valid_opt(f, c, domain) -> typing.Tuple[float, float, np.ndarray, np.ndarray]:
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(domain[0, 0], domain[0, 1], nx)
    y = np.linspace(domain[1, 0], domain[1, 1], ny)
    xv, yv = np.meshgrid(x, y)
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T

    true_values = f(samples)
    true_cond = c(samples)
    valid_data_idx = np.where(true_cond < LAMBDA)[0]
    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]
    f_max = np.max(np.abs(true_values))
    x_max = np.argmax(np.abs(true_values))

    return f_opt, f_max, x_opt, x_max


def train_on_toy(agent, iteration):
    print('Running model on toy example.')

    seed = 22
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    for j in range(iteration):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain_x.shape[0])

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(size=(x.shape[0],), scale=0.01)
        cost_val = c(x) + np.random.normal(size=(x.shape[0],), scale=0.005)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain_x.shape[0])
    assert check_in_domain(solution)

    # Compute regret
    f_opt, f_max, x_opt, x_max = get_valid_opt(f, c, domain_x)
    if c(solution) > 0.0:
        regret = 1
    else:
        regret = (f(solution) - f_opt) / f_max

    print(f'Optimal value: {f_opt}\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}')

    return agent


def main():
    agent = BO_algo()
    train_on_toy(agent, iteration=10)


if __name__ == "__main__":
    main()