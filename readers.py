import logging
from distrparams import MassDistributionParams

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class ReadingFileError(Exception):
    '''Class for errors of reading from file processing'''
    pass

#readers functions
def read_uniform_distribution_params(stream, modelKey, distribution_params_default: MassDistributionParams):
    M0_local = get_float_from_file_or_default(stream, "M0", distribution_params_default.M0)

    N_local = get_int_from_file_or_default(stream, "N", distribution_params_default.N)
    if N_local < 1e3 or N_local > 1e7: 
        N_local = distribution_params_default.N
        logging.warning(f"invalid N value. Using default value N = {N_local}")

    L_local = get_int_from_file_or_default(stream, "L", distribution_params_default.L)
    if N_local - L_local > 1e6 or N_local - L_local < 1e2:
        L_local = N_local - int(1e2)
        logging.warning(f"invalid L value. Using L = {L_local}")

    monte_carlo_experiments_number_local = get_int_from_file_or_default(stream, "monte_carlo_experiments_number", distribution_params_default.MonteCarloExperimentsNumber)

    return MassDistributionParams(
        model = modelKey,
        M0 = M0_local,
        M1 = distribution_params_default.M1,
        M2 = distribution_params_default.M2,
        beta = distribution_params_default.beta,
        x0 = distribution_params_default.x0,
        sigma = distribution_params_default.sigma,
        N = N_local,
        L = L_local,
        MonteCarloExperimentsNumber = monte_carlo_experiments_number_local
    )


def read_power_law_distribution_params(stream, modelKey, distribution_params_default: MassDistributionParams):
    M1_local = get_float_from_file_or_default(stream, "M1", distribution_params_default.M1)

    M2_local = get_float_from_file_or_default(stream, "M2", distribution_params_default.M2)
    
    beta_local = get_float_from_file_or_default(stream, "beta", distribution_params_default.beta)
    if beta_local < 2.0 or beta_local > 3.0:
        beta_local = distribution_params_default.beta
        logging.warning(f"invalid beta value. Using default value beta = {beta_local}")

    N_local = get_int_from_file_or_default(stream, "N", distribution_params_default.N)
    if N_local < 1e3 or N_local > 1e7: 
        N_local = distribution_params_default.N
        logging.warning(f"invalid N value. Using default value N = {N_local}")

    L_local = get_int_from_file_or_default(stream, "L", distribution_params_default.L)
    if N_local - L_local > 1e6 or N_local - L_local < 1e2:
        L_local = N_local - int(1e2)
        logging.warning(f"invalid L value. Using L = {L_local}")

    monte_carlo_experiments_number_local = get_int_from_file_or_default(stream, "monte_carlo_experiments_number", distribution_params_default.MonteCarloExperimentsNumber)

    return MassDistributionParams(
        model = modelKey,
        M0 = distribution_params_default.M0,
        M1 = M1_local,
        M2 = M2_local,
        beta = beta_local,
        x0 = distribution_params_default.x0,
        sigma = distribution_params_default.sigma,
        N = N_local,
        L = L_local,
        MonteCarloExperimentsNumber = monte_carlo_experiments_number_local
    )

def read_lognormal_law_distribution_params(stream, modelKey, distribution_params_default: MassDistributionParams):
    M1_local = get_float_from_file_or_default(stream, "M1", distribution_params_default.M1)

    M2_local = get_float_from_file_or_default(stream, "M2", distribution_params_default.M2)
    
    x0_local = get_float_from_file_or_default(stream, "x0", distribution_params_default.x0)

    sigma_local = get_float_from_file_or_default(stream, "sigma", distribution_params_default.sigma)

    N_local = get_int_from_file_or_default(stream, "N", distribution_params_default.N)
    if N_local < 1e3 or N_local > 1e7: 
        N_local = distribution_params_default.N
        logging.warning(f"invalid N value. Using default value N = {N_local}")

    L_local = get_int_from_file_or_default(stream, "L", distribution_params_default.L)
    if N_local - L_local > 1e6 or N_local - L_local < 1e2:
        L_local = N_local - int(1e2)
        logging.warning(f"invalid L value. Using L = {L_local}")

    monte_carlo_experiments_number_local = get_int_from_file_or_default(stream, "monte_carlo_experiments_number", distribution_params_default.MonteCarloExperimentsNumber)

    return MassDistributionParams(
        model = modelKey,
        M0 = distribution_params_default.M0,
        M1 = M1_local,
        M2 = M2_local,
        beta = distribution_params_default.beta,
        x0 = x0_local,
        sigma = sigma_local,
        N = N_local,
        L = L_local,
        MonteCarloExperimentsNumber = monte_carlo_experiments_number_local
    )


#dictionary for readers for delegate-like usage
model_params_reader = {
    "A": read_uniform_distribution_params,
    "B": read_power_law_distribution_params,
    "C": read_lognormal_law_distribution_params
}

def read_model_parameters(distribution_params_default: MassDistributionParams):
    try:
        with open("inputFile.txt", "r") as f:
            try:
                modelKey = read_line_data(f)
            except ReadingFileError as e:
                logging.critical(f"can't read model key: {e}")
                raise
        
            if modelKey not in model_params_reader:
                logging.critical(f"Incorrect model key: {modelKey}")
                raise ValueError(f"Incorrect model key: {modelKey}")
            
            return model_params_reader[modelKey](f, modelKey, distribution_params_default)
    except FileNotFoundError:
        logging.critical(f"Can't find input file.")
        raise ReadingFileError("file not found: inputFile.txt")



#internal functions for specific values reading
def read_line_data(stream):
    line = stream.readline()

    if not line:
        raise ReadingFileError("Unexpected end of file.")
    
    return line.split(":")[-1].strip()

def get_float_from_file_or_default(stream, param_name: str, default_value: float):
    try:
        data = read_line_data(stream)
    except ReadingFileError as e:
        result = default_value
        logging.warning(f"{param_name} wasn't read: {e}. Using default_value: {param_name} = {result}")
        return result
    
    try:
        result = float(data)
    except ValueError:
        result = default_value
        logging.warning(f"{param_name}: input file has wrong format. Using default_value: {param_name} = {result}")
    
    return result
    
def get_int_from_file_or_default(stream, param_name: str, default_value: int):
    try:
        data = read_line_data(stream)
    except ReadingFileError as e:
        result = default_value
        logging.warning(f"{param_name} wasn't read: {e}. Using default_value: {param_name} = {result}")
        return result
    
    try:
        result = int(data)
    except ValueError:
        result = default_value
        logging.warning(f"{param_name}: input file has wrong format. Using default_value: {param_name} = {result}")
    
    return result