import numpy as np
import torch


def H(x, eps=1e-6):
    """ Compute the element-wise entropy of x
    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)
    Keyword Arguments:
        eps {float} -- prevent failure on x == 0
    Returns:
        torch.Tensor -- H(x)
    """
    H = -(x+eps)*torch.log(x+eps)
    print('H', H)
    return H


def binary_entropy(x):
    """ Compute the element-wise binary entropy of 0 <= x <= 1.
        Avoid NaN for x = 0 and x = 1.
        Sets output to 0 for all values < 10^-10 or > 1-10^-10.
    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)
    Returns:
        H {torch.Tensor} -- binary entropy of x
    """
    x =  x * torch.logical_and(x >= 10**-10, x <= 1 - 10**-10) + (10**-11) * (x < 10**-10) + (1-10**-11) * (x > (1-10**-10))
    H = torch.zeros(x.size()) + ( -x * torch.log2(x) - (1 - x) * torch.log2(1-x)) * torch.logical_and(x >= 10**-10, x <= 1 - 10**-10)
    return H


def expected_entropy(mu, s2):
    """ Second part of mutual information.
        See also Houlsby et al. 2011.
        Arguments:
            mu {torch.Tensor} -- approximated posterior predictive mean at point of interest
            s2 {torch.Tensor} -- approximated posterior predictive variance at point of interest
        Returns:
            expH {torch.Tensor} -- H(x)
        """
    C = torch.sqrt(np.pi*torch.log(torch.tensor([2])) * 0.5)
    C2 = torch.pow(C, 2)
    s4 = torch.pow(s2, 2)
    expH = C / (s4 + C2) * torch.exp(-torch.pow(mu, 2) / (2 * (s4 + C2)))
    return expH


def move_s(sample, label, from_set, to_set):
    index = (from_set == sample).nonzero(as_tuple=True)[0]
    from_set = torch.cat([from_set[:index], from_set[index+1:]])
    to_set.inputs, _ = torch.sort(torch.cat([to_set.inputs, torch.Tensor([sample])]))
    index = (to_set.inputs == sample).nonzero(as_tuple=True)[0]
    if len(index) > 1:
        index = index[0]
    to_set.labels = torch.cat([to_set.labels[:index], label, to_set.labels[index:]])
    return from_set, to_set


def move_sample(sample, label, from_set, to_set):
    index = (from_set.inputs == sample).nonzero(as_tuple=True)[0]
    # label = torch.Tensor([from_set.labels[index]])
    from_set.inputs = torch.cat([from_set.inputs[:index], from_set.inputs[index+1:]])
    from_set.labels = torch.cat([from_set.labels[:index], from_set.labels[index+1:]])
    to_set.inputs, _ = torch.sort(torch.cat([to_set.inputs, torch.Tensor([sample])]))
    index = (to_set.inputs == sample).nonzero(as_tuple=True)[0]
    if len(index) > 1:
        index = index[0]
    #print('index', index)
    to_set.labels = torch.cat([to_set.labels[:index], label, to_set.labels[index:]])
    #print('to_set.inputs.numel', to_set.inputs.numel())
    #print('to_set.labels.numel', to_set.labels.numel())


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))