import numpy as np
from collections import namedtuple


def normal(x, mu, sigma):
    return np.exp(-((x - mu)**2) / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))

def lognormal(x, mu, sigma):
    return np.exp(-(((np.log(x) - mu) / sigma)**2) / 2.0) / (x * sigma * np.sqrt(2.0 * np.pi))

def exponential(x, a, b):
    return a * np.exp(-b * x)

ModelForFittingInfo = namedtuple("ModelForFittingInfo", ["Func", "ParamsNumber", "ParamsInitialGuess"])

models_for_fitting = {
    'normal': ModelForFittingInfo(normal, 2, [0.0, 1.0]),
    'lognormal': ModelForFittingInfo(lognormal, 2, [0.0, 1.0]),
    'exponential': ModelForFittingInfo(exponential, 2, [1.0, 1.0]),
}