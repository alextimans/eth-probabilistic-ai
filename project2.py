import os
import typing
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from tqdm import trange

from project2_util import ece, ParameterDistribution


class Model(object):

    def __init__(self):
        self.num_epochs = 5
        self.batch_size = 128
        learning_rate = 1e-3
        hidden_layers = (100, 100)
        use_densenet = False # Run a DenseNet for comparison
        self.print_interval = 100

        # Determine network type
        if use_densenet:
            print('Using a DenseNet model for comparison')
            self.network = DenseNet(in_features=28 * 28, hidden_features=hidden_layers, out_features=10)
        else:
            print('Using a BayesNet model')
            self.network = BayesNet(in_features=28 * 28, hidden_features=hidden_layers, out_features=10)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def train(self, dataset: torch.utils.data.Dataset):
        """
        If the network is a DenseNet, this performs normal stochastic gradient descent training.
        If the network is a BayesNet, this performs Bayes by Backprop.
        """

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.network.train()

        progress_bar = trange(self.num_epochs)
        for _ in progress_bar:
            num_batches = len(train_loader)
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):

                self.network.zero_grad()

                if isinstance(self.network, DenseNet):
                    # DenseNet training step

                    current_logits = self.network(batch_x)
                    loss = F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y, reduction='sum')
                    loss.backward()
                else:
                    assert isinstance(self.network, BayesNet)
                    # BayesNet training step

                    current_logits, log_prior, log_variational_posterior = self.network(batch_x)
                    nll_batch = F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y, reduction='sum')
                    loss = (1 / num_batches) * (log_variational_posterior - log_prior) + nll_batch
                    loss.backward()

                self.optimizer.step()

                # Update progress bar
                if batch_idx % self.print_interval == 0:
                    if isinstance(self.network, DenseNet):
                        current_logits = self.network(batch_x)
                    else:
                        assert isinstance(self.network, BayesNet)
                        current_logits, _, _ = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item())

    def predict(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Predict class probabilities using trained model.

        :data_loader: Data loader yielding the samples to predict on
        :return: (num_samples, 10) float array where second dimension sums up to 1 for each row
        """

        self.network.eval()

        probability_batches = []
        for batch_x, batch_y in data_loader:
            current_probabilities = self.network.predict_probabilities(batch_x).detach().numpy()
            probability_batches.append(current_probabilities)

        output = np.concatenate(probability_batches, axis=0)
        assert isinstance(output, np.ndarray)
        assert output.ndim == 2 and output.shape[1] == 10
        assert np.allclose(np.sum(output, axis=1), 1.0)

        return output


class BayesianLayer(nn.Module):
    """
    Implementing a single Bayesian feedforward layer.
    It maintains a prior and variational posterior for the weights and biases
    and uses sampling to approximate the gradients via Bayes by Backprop.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Weights prior
        self.prior = UnivariateGaussian(
            mu=torch.tensor(0.0),
            sigma=torch.tensor(1.0)
            )

        assert isinstance(self.prior, ParameterDistribution)
        assert not any(True for _ in self.prior.parameters()), 'Prior cannot have parameters'

        # Init variational posterior params as Gaussians with small values close to 0
        self.weights_var_posterior = MultivariateDiagonalGaussian(
            mu=nn.Parameter(
                torch.FloatTensor(self.out_features, self.in_features).normal_(mean=0, std=0.01)
                ),
            rho=nn.Parameter(
                torch.FloatTensor(self.out_features, self.in_features).normal_(mean=-2.5, std=0.01)
                )
            )

        assert isinstance(self.weights_var_posterior, ParameterDistribution)
        assert any(True for _ in self.weights_var_posterior.parameters()), 'Weight posterior must have parameters'

        if self.use_bias:
            self.bias_var_posterior = MultivariateDiagonalGaussian(
                mu=nn.Parameter(
                    torch.FloatTensor(self.out_features).normal_(mean=0, std=0.01)
                    ),
                rho=nn.Parameter(
                    torch.FloatTensor(self.out_features).normal_(mean=-2.5, std=0.01)
                    )
                )

            assert isinstance(self.bias_var_posterior, ParameterDistribution)
            assert any(True for _ in self.bias_var_posterior.parameters()), 'Bias posterior must have parameters'
        else:
            self.bias_var_posterior = None

    def forward(self, inputs: torch.Tensor):
        """
        Perform one forward pass through the layer with Bayes by Backprop.
        """

        eps = torch.randn(self.out_features, self.in_features)
        weights = self.weights_var_posterior.mu + F.softplus(
            self.weights_var_posterior.rho) * eps

        log_prior = self.prior.log_likelihood(weights).sum()
        log_variational_posterior = self.weights_var_posterior.log_likelihood(weights).sum()

        if self.use_bias:
            eps = torch.randn(self.out_features)
            bias = self.bias_var_posterior.mu + F.softplus(
                self.bias_var_posterior.rho) * eps

            log_prior += self.prior.log_likelihood(bias).sum()
            log_variational_posterior += self.bias_var_posterior.log_likelihood(bias)
        else:
            bias = None

        return F.linear(inputs, weights, bias), log_prior, log_variational_posterior


class BayesNet(nn.Module):
    """
    Implementing a Bayesian feedforward neural network using BayesianLayer objects.
    """

    def __init__(self, in_features: int, hidden_features: typing.Tuple[int, ...], out_features: int):
        super().__init__()

        feature_sizes = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(feature_sizes) - 1
        self.layers = nn.ModuleList([
            BayesianLayer(feature_sizes[idx], feature_sizes[idx + 1], bias=True)
            for idx in range(num_affine_maps)
        ])
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one forward pass through the BNN using a single set of weights
        sampled from the variational posterior.
        """

        current_features = x
        log_prior = torch.tensor(0.0)
        log_variational_posterior = torch.tensor(0.0)

        for idx, current_layer in enumerate(self.layers):
            new_features, log_prior_layer, log_var_posterior_layer = current_layer(current_features)
            if idx < len(self.layers) - 1:
                new_features = self.activation(new_features)

            current_features = new_features
            log_prior += log_prior_layer
            log_variational_posterior += log_var_posterior_layer

        output_features = current_features

        return output_features, log_prior, log_variational_posterior

    def predict_probabilities(self, x: torch.Tensor, num_mc_samples: int = 10) -> torch.Tensor:
        """
        Predict class probabilities for the given features by sampling from the BNN.

        :num_mc_samples: Number of MC samples to take for prediction
        :return: Predicted class probabilities, float tensor of shape (batch_size, 10)
            such that the last dimension sums up to 1 for each row
        """
        probability_samples = torch.stack([F.softmax(self.forward(x)[0], dim=1) for _ in range(num_mc_samples)], dim=0)
        estimated_probability = torch.mean(probability_samples, dim=0)

        assert estimated_probability.shape == (x.shape[0], 10)
        assert torch.allclose(torch.sum(estimated_probability, dim=1), torch.tensor(1.0))

        return estimated_probability


class UnivariateGaussian(ParameterDistribution):

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super(UnivariateGaussian, self).__init__()
        assert mu.size() == () and sigma.size() == ()
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return Normal(self.mu, self.sigma).log_prob(values)

    def sample(self) -> torch.Tensor:
        return Normal(self.mu, self.sigma).sample()


class MultivariateDiagonalGaussian(ParameterDistribution):
    """
    Multivariate diagonal Gaussian distribution.
    Parameterizes the standard deviation via params rho.
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super(MultivariateDiagonalGaussian, self).__init__()
        assert mu.size() == rho.size()
        self.mu = mu
        self.rho = rho

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.rho)
        return Independent(Normal(self.mu, sigma), 1
                           ).log_prob(values)

    def sample(self) -> torch.Tensor:
        sigma = F.softplus(self.rho)
        return Independent(Normal(self.mu, sigma), 1
                           ).sample(self.mu.size())


def evaluate(model: Model, eval_loader: torch.utils.data.DataLoader, data_dir: str, output_dir: str):

    # Predict class probabilities on test data
    predicted_probabilities = model.predict(eval_loader)

    # Calculate evaluation metrics
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    actual_classes = eval_loader.dataset.tensors[1].detach().numpy()
    accuracy = np.mean((predicted_classes == actual_classes))
    ece_score = ece(predicted_probabilities, actual_classes)
    print(f'Accuracy: {accuracy.item():.3f}, ECE score: {ece_score:.3f}')


class DenseNet(nn.Module):
    """
    Implementing a feedforward neural network.
    Reference/baseline for calibration in the normal neural network case.
    """

    def __init__(self, in_features: int, hidden_features: typing.Tuple[int, ...], out_features: int):
        super().__init__()

        feature_sizes = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(feature_sizes) - 1
        self.layers = nn.ModuleList([
            nn.Linear(feature_sizes[idx], feature_sizes[idx + 1], bias=True)
            for idx in range(num_affine_maps)
        ])
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_features = x

        for idx, current_layer in enumerate(self.layers):
            new_features = current_layer(current_features)
            if idx < len(self.layers) - 1:
                new_features = self.activation(new_features)
            current_features = new_features

        return current_features

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 28 ** 2
        estimated_probability = F.softmax(self.forward(x), dim=1)
        assert estimated_probability.shape == (x.shape[0], 10)

        return estimated_probability


def plot_data(dtrain, fig_size, grid_size):
    figure = plt.figure(figsize=(fig_size, fig_size))
    cols, rows = grid_size, grid_size

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dtrain), size=(1, )).item()
        img, label = dtrain[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title('Label: %i' %label.item())
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def main():
    # Load training data
    data_dir = os.curdir
    output_dir = os.curdir
    raw_train_data = np.load(os.path.join(data_dir, 'train_data.npz'))

    x_train = torch.from_numpy(raw_train_data['train_x'])
    y_train = torch.from_numpy(raw_train_data['train_y']).long()
    plot_data(torch.utils.data.TensorDataset(x_train, y_train), 8, 3) # Visualize data
    x_train = x_train.reshape([-1, 784])
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

    model = Model()

    print('Training model')
    model.train(dataset_train)

    print('Evaluating model on training data')
    eval_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=64, shuffle=False, drop_last=False
    )
    evaluate(model, eval_loader, data_dir, output_dir)


if __name__ == "__main__":
    main()