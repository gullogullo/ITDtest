import numpy as np
from scipy.stats import norm
import torch

from util import *


class Acquirer:
    """ Base class for acquisition function
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def score(model, likelihood, x):
        """ Parallezied acquisition scoring function
        Arguments:
            model -- the model
            x -- datapoints to evaluate
        Returns:
            [torch.Tensor] -- a vector of acquisition scores
        """
        return torch.zeros(len(x))
    
    def select_samples(self, model, likelihood, pool_data):
        # score every datapoint in the pool under the model
        # initialize scores tensor
        scores = torch.zeros(len(pool_data))
        # compute the scores with the model
        scores = self.score(model, likelihood, pool_data)
        # return the indices sorted by score, or the argmax
        # print('scores', scores)
        best_local_index = torch.argsort(scores) #[-1]
        best_local_index = best_local_index[-1]
        # return the indices sorted by score of the pool, or the argmax
        # best_global_index = np.array(pool_data.indices)[best_local_index.cpu().numpy()]
        best_sample = pool_data[best_local_index.cpu().numpy()]
        #print('best_sample', best_sample)
        #print('best_local_index', best_local_index)
        return best_sample


class BALD(Acquirer):
    def __init__(self, pool_data):
        super(BALD, self).__init__(pool_data)

    @staticmethod
    def score(model, likelihood, x):
        with torch.no_grad():
            y_preds = likelihood(model(x))
            mu_x = y_preds.mean.float()
            s2_x = y_preds.variance.float()
            phi = norm.cdf(mu_x / torch.sqrt(torch.pow(s2_x, 2) + 1))
            H = binary_entropy(torch.from_numpy(phi))
            # print('H', H)
            expH = expected_entropy(mu_x, s2_x)
            # print('expH', expH)
            return H - expH


class Random(Acquirer):
    def __init__(self, pool_data):
        super(Random, self).__init__(pool_data)

    @staticmethod
    def score(model, likelihood, x):
        return torch.Tensor(np.array(np.random.rand(len(x))))
