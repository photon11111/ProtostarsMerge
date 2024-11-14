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

def generate_power_law_distribution_smoothed(distribution_params):
    uniform_distr = np.random.uniform(low=0.0, high=1.0, size=distribution_params.N)
    masses = [(1.0 - x)**(1.0 / (1.0 - distribution_params.beta)) for x in uniform_distr]
    x, y = analyse_sample(masses)
    result_x, result_y = smooth_distribution(x, y, 1.0)
    return result_x

def generate_lognormal_law_distribution_smoothed(distribution_params):
    #uniform_distr = np.random.uniform(low=0.0, high=1.0, size=distribution_params.N)
    #return [np.exp(distribution_params.x0 + np.sqrt(2.0) * distribution_params.sigma * erfinv(2.0 * x - 1)) for x in uniform_distr]
    masses =  [x for x in np.random.lognormal(distribution_params.x0, distribution_params.sigma, distribution_params.N)]
    x, y = analyse_sample(masses)
    result_x, result_y = smooth_distribution(x, y)
    return result_x

#dictionary for generators for delegate-like usage
model_distribution_generators = {
    "A": generate_uniform_distribution,
    "B": generate_power_law_distribution,
    "C": generate_lognormal_law_distribution
}

def analyse_sample(masses, accuracy = 1e-5):
    masses.sort()
    mass_values = []
    mass_counts = []

    i, j = 0, 0

    while i < len(masses):
        while j < len(masses) and np.abs(masses[j] - masses[i]) < accuracy:
            j += 1

        mass_values.append(masses[i])
        mass_counts.append(j - i)
        i = j

    return mass_values, mass_counts

def smooth_distribution(mass_values, mass_counts, ddx=0.1):
    results_x = []
    results_y = []

    results_x.append(mass_values[0])
    results_y.append(mass_counts[0])

    for i in range(1, len(mass_values)):
        dx = mass_values[i] - mass_values[i - 1]
        dy = mass_counts[i] - mass_counts[i - 1]

        results_x.append(mass_values[i])
        results_y.append(mass_counts[i])

        if dx > 2.0 * ddx:
            n = int(dx / ddx)  
            ddy = dy / n       

            for j in range(1, n):
                interpolated_x = mass_values[i - 1] + j * ddx
                interpolated_y = mass_counts[i - 1] + j * ddy
                results_x.append(interpolated_x)
                results_y.append(interpolated_y)

    return results_x, results_y