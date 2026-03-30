import numpy as np
import pytest
from classical_laminate_theory.failuremodels import Puck, Hashin, FailureMode
from classical_laminate_theory.structures import Material

# Material properties for a typical GRP (Glass Reinforced Plastic)
GRP_PROPS = {
    "Xt": 1000e6,
    "Xc": 800e6,
    "Yt": 40e6,
    "Yc": 120e6,
    "S12": 40e6,
    "E1": 40e9,
    "E2": 10e9,
    "G12": 5e9,
    "v12": 0.25
}

@pytest.fixture
def grp_material():
    return Material(**GRP_PROPS)

@pytest.fixture
def puck_model(grp_material):
    return Puck(material=grp_material)

@pytest.fixture
def hashin_model(grp_material):
    return Hashin(material=grp_material)

# --- PUCK VALIDATION ---

def test_puck_pure_tension_ff(puck_model):
    """Puck FF: Verify failure at exactly Xt (Eq 1)."""
    stress = np.array([GRP_PROPS["Xt"] * 1.01, 0, 0])
    strain = np.array([stress[0] / GRP_PROPS["E1"], 0, 0])
    modes = puck_model.failure_check(stress, strain)
    assert FailureMode.FIBER_TENSION in modes

def test_puck_pure_compression_ff(puck_model):
    """Puck FF: Verify failure at exactly Xc (Eq 2)."""
    stress = np.array([-GRP_PROPS["Xc"] * 1.01, 0, 0])
    strain = np.array([stress[0] / GRP_PROPS["E1"], 0, 0])
    modes = puck_model.failure_check(stress, strain)
    assert FailureMode.FIBER_COMPRESSION in modes

def test_puck_mode_a_tension(puck_model):
    """Puck IFF Mode A: Pure transverse tension (sigma2 = Yt)."""
    stress = np.array([0, GRP_PROPS["Yt"] * 1.01, 0])
    modes = puck_model.failure_check(stress, np.zeros(3))
    assert FailureMode.MATRIX_TENSION in modes

def test_puck_pure_shear(puck_model):
    """Puck IFF: Pure shear failure (tau12 = S12)."""
    stress = np.array([0, 0, GRP_PROPS["S12"] * 1.01])
    modes = puck_model.failure_check(stress, np.zeros(3))
    assert FailureMode.MATRIX_TENSION in modes or FailureMode.MATRIX_COMPRESSION in modes

def test_puck_mode_c_compression(puck_model):
    """Puck IFF Mode C: High transverse compression (sigma2 >> S12)."""
    stress = np.array([0, -GRP_PROPS["Yc"] * 1.01, 0])
    modes = puck_model.failure_check(stress, np.zeros(3))
    assert FailureMode.MATRIX_COMPRESSION in modes

# --- HASHIN VALIDATION ---

def test_hashin_fiber_tension(hashin_model):
    """Hashin FF: Fiber tension with shear interaction."""
    stress = np.array([0.8 * GRP_PROPS["Xt"], 0, 0.8 * GRP_PROPS["S12"]])
    modes = hashin_model.failure_check(stress, np.zeros(3))
    assert FailureMode.FIBER_TENSION in modes

def test_hashin_matrix_tension(hashin_model):
    """Hashin IFF: Matrix tension with shear interaction."""
    stress = np.array([0, 0.8 * GRP_PROPS["Yt"], 0.8 * GRP_PROPS["S12"]])
    modes = hashin_model.failure_check(stress, np.zeros(3))
    assert FailureMode.MATRIX_TENSION in modes

def test_hashin_matrix_compression(hashin_model):
    """Hashin IFF: Matrix compression."""
    stress = np.array([0, -GRP_PROPS["Yc"] * 1.01, 0])
    modes = hashin_model.failure_check(stress, np.zeros(3))
    assert FailureMode.MATRIX_COMPRESSION in modes

# --- BASIC MODELS VALIDATION ---

def test_max_stress():
    """Verify simple MaxStress logic."""
    from classical_laminate_theory.failuremodels import MaxStress
    model = MaxStress(Xt=100, Xc=100, Yt=50, Yc=50, S12=20)
    
    assert FailureMode.FIBER_TENSION in model.failure_check([101, 0, 0], [0, 0, 0])
    assert FailureMode.SHEAR in model.failure_check([0, 0, 21], [0, 0, 0])
    assert not model.failure_check([99, 49, 19], [0, 0, 0])

def test_tsai_hill():
    """Verify Tsai-Hill interactive criterion."""
    from classical_laminate_theory.failuremodels import TsaiHill
    model = TsaiHill(X11=100, X22=50, S12=20)
    
    assert FailureMode.GENERAL_FAILURE in model.failure_check([101, 0, 0], [0, 0, 0])
    assert FailureMode.GENERAL_FAILURE in model.failure_check([90, 45, 0], [0, 0, 0])
