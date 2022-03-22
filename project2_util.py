import abc
import warnings
import numpy as np
import torch


def ece(predicted_probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 30) -> float:
    """
    Computes the Expected Calibration Error (ECE).

    :n_bins: Number of bins for histogram binning
    :return: ECE score as a float
    """

    num_samples, num_classes = predicted_probabilities.shape

    predictions = np.argmax(predicted_probabilities, axis=1)
    prediction_confidences = predicted_probabilities[range(num_samples), predictions]

    # Use uniform bins on the range of probabilities, i.e. closed interval [0.,1.]
    bin_upper_edges = np.histogram_bin_edges([], bins=n_bins, range=(0., 1.))
    bin_upper_edges = bin_upper_edges[1:]

    probs_as_bin_num = np.digitize(prediction_confidences, bin_upper_edges)
    sums_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins, weights=prediction_confidences)
    sums_per_bin = sums_per_bin.astype(np.float32)

    total_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins) \
        + np.finfo(sums_per_bin.dtype).eps
    avg_prob_per_bin = sums_per_bin / total_per_bin

    onehot_labels = np.eye(num_classes)[labels]
    accuracies = onehot_labels[range(num_samples), predictions]
    accuracies_per_bin = np.bincount(probs_as_bin_num, weights=accuracies, minlength=n_bins) / total_per_bin

    prob_of_being_in_a_bin = total_per_bin / float(num_samples)

    ece_ret = np.abs(accuracies_per_bin - avg_prob_per_bin) * prob_of_being_in_a_bin
    ece_ret = np.sum(ece_ret)

    return float(ece_ret)


class ParameterDistribution(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract class that models a distribution over model parameters,
    usable for Bayes by Backprop.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        warnings.warn('ParameterDistribution should not be called! Use its explicit methods!')
        return self.log_likelihood(values)
