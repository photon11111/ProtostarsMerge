from dataclasses import dataclass

@dataclass
class MassDistributionParams:
    model: str
    M0: float
    beta: float
    x0: float
    sigma: float
    N: int
    L: int
    MonteCarloExperimentsNumber: int