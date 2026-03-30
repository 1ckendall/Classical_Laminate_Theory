import numpy as np
import pytest
from classical_laminate_theory.structures import Laminate, Material
from classical_laminate_theory.failuremodels import MaxStress
from classical_laminate_theory.progressive_failure_analysis import ProgressiveFailureAnalysis

@pytest.fixture
def pfa_setup():
    props = {
        "E1": 140e9, "E2": 10e9, "G12": 5e9, "v12": 0.3, "t": 0.15e-3,
        "Xt": 1000e6, "Xc": 800e6, "Yt": 40e6, "Yc": 120e6, "S12": 40e6
    }
    mat = Material(**props)
    # Link model to material
    model = MaxStress(material=mat)
    # Assemble laminate from model
    lam = Laminate.from_layup("[0/90]_s", failure_model=model)
    return ProgressiveFailureAnalysis(lam)

def test_pfa_simulation_generator(pfa_setup):
    """Verify that simulate generator yields expected state."""
    load_path = [np.array([f * 1e6, 0, 0, 0, 0, 0]) for f in np.linspace(0, 1.0, 5)]
    states = list(pfa_setup.simulate(load_path))
    assert len(states) > 0
    assert "load" in states[0]
    assert "is_intact" in states[0]

def test_pfa_rupture_criteria(pfa_setup):
    """Test that the pfa correctly detects rupture under massive load."""
    massive_load = np.array([5000e6, 0, 0, 0, 0, 0])
    load_path = [massive_load]
    states = list(pfa_setup.simulate(load_path))
    assert states[-1]["is_intact"] is False
    assert states[-1]["stiffness_ratio"] < 0.1

def test_pfa_non_destructive_reset(pfa_setup):
    """Confirm PFA doesn't permanently damage the original laminate."""
    initial_a11 = pfa_setup.laminate.A[0, 0]
    pfa_setup.run_until_failure(np.array([1, 0, 0, 0, 0, 0]), step_size=5e5)
    pfa_setup.laminate.reset_damage()
    assert np.isclose(pfa_setup.laminate.A[0, 0], initial_a11)
