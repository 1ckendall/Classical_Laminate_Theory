import numpy as np

from classical_laminate_theory.failuremodels import TsaiHill
from classical_laminate_theory.helpers import laminate_builder
from classical_laminate_theory.structures import Lamina, Laminate

failure_model = TsaiHill(100e6, 10e6, 30e6)
L = laminate_builder(
    "[90, [+-63.4]_2s, 90, 90]",
    1e9,
    0.1e9,
    0.3e9,
    0.4,
    0.1e-3,
    failure_model,
    loading=np.array([1e5, 2e5, 0, 0, 0, 0]),
)
L.global_stress_graph
L.local_stress_graph
