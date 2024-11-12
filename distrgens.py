from distrparams import MassDistributionParams
import numpy as np
from scipy.special import erfinv

#model mass generators
def generate_uniform_distribution(distribution_params):
    return [x for x in np.full(distribution_params.N, distribution_params.M0)]

def generate_power_law_distribution(distribution_params):
    uniform_distr = np.random.uniform(low=0.0, high=1.0, size=distribution_params.N)
    return [(1.0 - x)**(1.0 / (1.0 - distribution_params.beta)) for x in uniform_distr]

def generate_lognormal_law_distribution(distribution_params):
    #uniform_distr = np.random.uniform(low=0.0, high=1.0, size=distribution_params.N)
    #return [np.exp(distribution_params.x0 + np.sqrt(2.0) * distribution_params.sigma * erfinv(2.0 * x - 1)) for x in uniform_distr]
    return [x for x in np.random.lognormal(distribution_params.x0, distribution_params.sigma, distribution_params.N)]

#dictionary for generators for delegate-like usage
model_distribution_generators = {
    "A": generate_uniform_distribution,
    "B": generate_power_law_distribution,
    "C": generate_lognormal_law_distribution
}