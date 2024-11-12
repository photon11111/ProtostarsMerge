import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import logging
import multiprocessing as mp
from distrparams import MassDistributionParams
from distrgens import model_distribution_generators
from readers import read_model_parameters
from fittingmodels import models_for_fitting


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def merge_stars(masses):
    if (len(masses) < 2):
        return
    
    i, j = random.sample(range(len(masses)), 2)

    masses[i] += masses[j]
    masses[j] = masses[-1]
    masses.pop()

def run_experiment(distribution_params):
    masses = model_distribution_generators[distribution_params.model](distribution_params)

    for _ in range(distribution_params.L):
        merge_stars(masses)
        
    return masses

def monte_carlo_simulation(distribution_params, num_workers=4):
    final_masses = []
    
    with mp.Pool(num_workers) as pool:
        results = [pool.apply_async(run_experiment, (distribution_params,)) for _ in range(distribution_params.MonteCarloExperimentsNumber)]
        
        for r in results:
            final_masses.extend(r.get())

    return final_masses

def calculate_aic(n, rss, num_params):
    return 2 * num_params + n * np.log(rss / n)

def analyse_sample(masses, accuracy = 1e-5):
    masses.sort()
    mass_values = []
    mass_probabilities = []

    i, j = 0, 0

    while i < len(masses):
        while j < len(masses) and np.abs(masses[j] - masses[i]) < accuracy:
            j += 1

        mass_values.append(masses[i])
        mass_probabilities.append((j - i) / len(masses))
        i = j

    return mass_values, mass_probabilities

def get_best_model(masses):
    x, y = analyse_sample(masses)

    sigma = np.ones_like(y)
    sigma[np.abs(y) < 1e-20] = np.inf

    best_model = None
    best_aic = np.inf
    best_popt = None

    for name, model_info in models_for_fitting.items():
        try:
            popt, _ = curve_fit(model_info.Func, x, y) #p0=model_info.ParamsInitialGuess)
            residuals = []
            for i  in range(len(y)):
                residuals.append(y[i] - model_info.Func(x[i], *popt))
            #residuals = y - model_info.Func(x, *popt)
            rss = np.sum([res**2 for res in residuals])
            aic = calculate_aic(len(x), rss, model_info.ParamsNumber)
            print(f"{name} model aic: {aic}")

            if aic < best_aic:
                best_aic = aic
                best_model = name
                best_popt = popt
        except Exception as e:
            logging.warning(f"fitting for model {name} was not successful: {e}")

    return best_model, best_popt, x, y

def plot_model(model, popt, x, y, pdf):
    model_func = models_for_fitting[model].Func
    y_fit = [model_func(arg, *popt) for arg in x]

    plt.figure()
    plt.bar(x, y, color='blue', label='Histogram')
    plt.plot(x, y_fit, 'r-', label=f'fitted on model: {model}')
    plt.xlabel('mass')
    plt.ylabel('probability density')
    plt.legend()
    pdf.savefig()
    plt.close()





#main code
distribution_params_default = MassDistributionParams(
    model = "",
    M0 = 1.0,
    M1 = 0.1,
    M2 = 10.0,
    beta = 2.35,
    x0 = 0.0,
    sigma = 1.0,
    N = 1000,
    L = 10,
    MonteCarloExperimentsNumber = 10000
)

distribution_params = read_model_parameters(distribution_params_default)

final_masses = monte_carlo_simulation(distribution_params)
model, popt, hist_x, hist_y = get_best_model(final_masses)

pdf = PdfPages("outputFile.pdf")
plot_model(model, popt, hist_x, hist_y, pdf)
pdf.close()