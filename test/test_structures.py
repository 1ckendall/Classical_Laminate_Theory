import numpy as np
import pytest
from classical_laminate_theory.structures import Lamina, Laminate, Material
from classical_laminate_theory.failuremodels import MaxStress

@pytest.fixture
def mat():
    return Material(E1=140e9, E2=10e9, G12=5e9, v12=0.3, t=0.15e-3)

def test_lamina_q_matrix(mat):
    """Verify local reduced stiffness matrix Q."""
    model = MaxStress(material=mat)
    ply = Lamina(angle=0, material=mat, failure_model=model)
    
    v21 = mat.v12 * mat.E2 / mat.E1
    expected_q11 = mat.E1 / (1 - mat.v12 * v21)
    
    assert np.isclose(ply.Qmat[0, 0], expected_q11)

def test_lamina_damage_reactive(mat):
    """Verify that changing damage factors updates stiffness properties."""
    ply = Lamina(angle=0, material=mat)
    
    initial_q11 = ply.Qmat[0, 0]
    ply.d1 = 0.5
    
    assert np.isclose(ply.E1, mat.E1 * 0.5)
    assert ply.Qmat[0, 0] < initial_q11
    
    ply.reset_damage()
    assert np.isclose(ply.Qmat[0, 0], initial_q11)

def test_laminate_symmetry_b_matrix(mat):
    """Verify that a symmetric layup has B=0."""
    # Using new Material-aware assembly
    lam = Laminate.from_layup("[0/90]_s", material=mat)
    assert np.allclose(lam.B, 0, atol=1e-9)

def test_laminate_z_positions(mat):
    """Verify ply boundary calculations."""
    lam = Laminate.from_layup("[0/90]_s", material=mat)
    
    total_thickness = 4 * mat.extra['t']
    assert np.isclose(lam.lamina_boundaries[-1] - lam.lamina_boundaries[0], total_thickness)
    assert np.isclose(np.mean(lam.lamina_boundaries), 0, atol=1e-12)
