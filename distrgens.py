from distrparams import MassDistributionParams
import numpy as np

#model mass generators
def generate_uniform_distribution(distribution_params):
    return [x for x in np.full(distribution_params.N, distribution_params.M0)]

def generate_power_law_distribution(distribution_params):
    return [x * (distribution_params.M2 - distribution_params.M1) + distribution_params.M1
            for x in np.random.power(a=abs(-distribution_params.beta), size=distribution_params.N)]

def generate_lognormal_law_distribution(distribution_params):
    masses = np.random.lognormal(distribution_params.x0, distribution_params.sigma, distribution_params.N)
    masses_normalized = (masses - np.min(masses)) / (np.max(masses) - np.min(masses))
    return [distribution_params.M1 + (distribution_params.M2 - distribution_params.M1) * mass for mass in masses_normalized]

#dictionary for generators for delegate-like usage
model_distribution_generators = {
    "A": generate_uniform_distribution,
    "B": generate_power_law_distribution,
    "C": generate_lognormal_law_distribution
}