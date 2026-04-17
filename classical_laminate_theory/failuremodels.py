from abc import ABC, abstractmethod
import numpy as np
from enum import Enum, auto


class FailureMode(Enum):
    """
    Enumeration of all possible composite failure modes.
    Used to signal exactly what broke so the PFA loop degrades the correct stiffness.
    """

    # Safe / No Failure
    SAFE = auto()

    # Fiber Dominated
    FIBER_TENSION = auto()
    FIBER_COMPRESSION = auto()

    # Matrix Dominated
    MATRIX_TENSION = auto()
    MATRIX_COMPRESSION = auto()
    SHEAR = auto()  # Pure shear (rarely distinct from matrix, but useful for MaxStress)

    # Generic / Combined (For Tsai-Hill)
    GENERAL_FAILURE = auto()


class FailureModel(ABC):
    def __init__(self, material=None):
        self.material = material

    @abstractmethod
    def failure_check(self, local_stress, local_strain) -> list[FailureMode]:
        """Returns a list of FailureMode enums."""
        raise NotImplementedError

    @abstractmethod
    def get_effort(self, local_stress, local_strain) -> float:
        """Returns the maximum failure index (effort). 1.0 = Failure."""
        raise NotImplementedError


class MaxStress(FailureModel):
    """
    Maximum Stress Failure Criterion.
    Checks each stress component independently against its respective strength.
    """

    def __init__(self, Xt=0, Xc=0, Yt=0, Yc=0, S12=0, material=None):
        super().__init__(material)
        self.Xt = Xt if Xt != 0 else (material.Xt if material else 0)
        self.Xc = Xc if Xc != 0 else (material.Xc if material else 0)
        self.Yt = Yt if Yt != 0 else (material.Yt if material else 0)
        self.Yc = Yc if Yc != 0 else (material.Yc if material else 0)
        self.S12 = S12 if S12 != 0 else (material.S12 if material else 0)

    def failure_check(self, local_stress, local_strain):
        s1, s2, t12 = local_stress
        failures = []

        # 1. Longitudinal (Fiber)
        if s1 >= 0:
            if s1 / self.Xt >= 1:
                failures.append(FailureMode.FIBER_TENSION)
        else:
            if abs(s1) / self.Xc >= 1:
                failures.append(FailureMode.FIBER_COMPRESSION)

        # 2. Transverse (Matrix)
        if s2 >= 0:
            if s2 / self.Yt >= 1:
                failures.append(FailureMode.MATRIX_TENSION)
        else:
            if abs(s2) / self.Yc >= 1:
                failures.append(FailureMode.MATRIX_COMPRESSION)

        # 3. Shear
        if abs(t12) / self.S12 >= 1:
            failures.append(FailureMode.SHEAR)

        return failures

    def get_effort(self, local_stress, local_strain):
        s1, s2, t12 = local_stress
        efforts = [
            s1 / self.Xt if s1 >= 0 else abs(s1) / self.Xc,
            s2 / self.Yt if s2 >= 0 else abs(s2) / self.Yc,
            abs(t12) / self.S12,
        ]
        return max(efforts)


class TsaiHill(FailureModel):
    def __init__(self, Xt=0, Xc=0, Yt=0, Yc=0, S12=0, material=None):
        super().__init__(material)
        self.Xt = Xt if Xt != 0 else (material.Xt if material else 0)
        self.Xc = Xc if Xc != 0 else (material.Xc if material else 0)
        self.Yt = Yt if Yt != 0 else (material.Yt if material else 0)
        self.Yc = Yc if Yc != 0 else (material.Yc if material else 0)
        self.S12 = S12 if S12 != 0 else (material.S12 if material else 0)

    def failure_check(self, local_stress, local_strain):
        if self.get_effort(local_stress, local_strain) >= 1:
            return [FailureMode.GENERAL_FAILURE]
        return []

    def get_effort(self, local_stress, local_strain):
        s1, s2, t12 = local_stress
        # Use tension or compression strength depending on stress sign
        X1 = self.Xt if s1 >= 0 else self.Xc
        X2 = self.Yt if s2 >= 0 else self.Yc
        return (
            (s1 / X1) ** 2
            - (s1 * s2) / (X1 ** 2)
            + (s2 / X2) ** 2
            + (t12 / self.S12) ** 2
        )


class Hashin(FailureModel):
    """
    Hashin Failure Criterion (2D Plane Stress).
    Distinguishes between fiber and matrix failure modes.
    """

    def __init__(self, Xt=0, Xc=0, Yt=0, Yc=0, S12=0, material=None):
        super().__init__(material)
        self.Xt = Xt if Xt != 0 else (material.Xt if material else 0)
        self.Xc = Xc if Xc != 0 else (material.Xc if material else 0)
        self.Yt = Yt if Yt != 0 else (material.Yt if material else 0)
        self.Yc = Yc if Yc != 0 else (material.Yc if material else 0)
        self.S12 = S12 if S12 != 0 else (material.S12 if material else 0)

    def failure_check(self, local_stress, local_strain):
        s1, s2, t12 = local_stress
        failures = []

        # 1. Fiber Tension (s1 >= 0)
        if s1 >= 0:
            if (s1 / self.Xt) ** 2 + (t12 / self.S12) ** 2 >= 1:
                failures.append(FailureMode.FIBER_TENSION)
        # 2. Fiber Compression (s1 < 0)
        else:
            if (abs(s1) / self.Xc) ** 2 >= 1:
                failures.append(FailureMode.FIBER_COMPRESSION)

        # 3. Matrix Tension (s2 >= 0)
        if s2 >= 0:
            if (s2 / self.Yt) ** 2 + (t12 / self.S12) ** 2 >= 1:
                failures.append(FailureMode.MATRIX_TENSION)
        # 4. Matrix Compression (s2 < 0)
        else:
            term1 = (s2 / (2 * self.S12)) ** 2
            term2 = ((self.Yc / (2 * self.S12)) ** 2 - 1) * (s2 / self.Yc)
            term3 = (t12 / self.S12) ** 2
            if term1 + term2 + term3 >= 1:
                failures.append(FailureMode.MATRIX_COMPRESSION)

        return failures

    def get_effort(self, local_stress, local_strain):
        s1, s2, t12 = local_stress
        # We calculate the indices for each quadrant
        indices = []
        if s1 >= 0:
            indices.append((s1 / self.Xt) ** 2 + (t12 / self.S12) ** 2)
        else:
            indices.append((abs(s1) / self.Xc) ** 2)

        if s2 >= 0:
            indices.append((s2 / self.Yt) ** 2 + (t12 / self.S12) ** 2)
        else:
            term1 = (s2 / (2 * self.S12)) ** 2
            term2 = ((self.Yc / (2 * self.S12)) ** 2 - 1) * (s2 / self.Yc)
            term3 = (t12 / self.S12) ** 2
            indices.append(term1 + term2 + term3)

        return max(indices)


class Puck(FailureModel):
    """
    Puck Failure Criterion (2D Plane Stress).
    Refactored to return FailureMode Enums for robust PFA integration.
    """

    def __init__(
        self,
        Xt=0,
        Xc=0,
        Yt=0,
        Yc=0,
        S12=0,
        E1=0,
        v12=0,
        m_sigF=1.1,
        p12_plus=0.3,
        p12_minus=0.2,
        weakening=0.1,
        material=None,
    ):
        super().__init__(material)
        # Pull from material if provided, otherwise use args
        self.Xt = Xt if Xt != 0 else (material.Xt if material else 0)
        self.Xc = Xc if Xc != 0 else (material.Xc if material else 0)
        self.Yt = Yt if Yt != 0 else (material.Yt if material else 0)
        self.Yc = Yc if Yc != 0 else (material.Yc if material else 0)
        self.S = S12 if S12 != 0 else (material.S12 if material else 0)
        self.E1 = E1 if E1 != 0 else (material.E1 if material else 0)
        self.v12 = v12 if v12 != 0 else (material.v12 if material else 0)

        self.m_sigF = m_sigF
        self.p12_plus = p12_plus
        self.p12_minus = p12_minus

        # Weakening coefficient (f_w influence)
        if weakening == "puck":
            self.weakening_coeff = 1.0 / 1.1
        elif weakening is None:
            self.weakening_coeff = 0.0
        else:
            self.weakening_coeff = float(weakening)

        # --- Pre-calculated Puck Constants ---
        # Guard against zero E1 during initialization
        self.eps_1T = self.Xt / self.E1 if self.E1 != 0 else 0
        self.eps_1C = self.Xc / self.E1 if self.E1 != 0 else 0

        # Derived fracture parameters
        if self.S != 0:
            term_root = 1 + 2 * self.p12_minus * (self.Yc / self.S)
            self.R_tt_A = (self.S / (2 * self.p12_minus)) * (np.sqrt(term_root) - 1)
            self.p23_minus = self.p12_minus * (self.R_tt_A / self.S)
            self.tau_12_c = self.S * np.sqrt(1 + 2 * self.p23_minus)
        else:
            self.R_tt_A = self.p23_minus = self.tau_12_c = 0

    def failure_check(self, local_stress, local_strain) -> list[FailureMode]:
        s1, s2, t12 = local_stress
        e1 = local_strain[0]

        failures = []

        # ---------------------------------------------------------
        # 1. Fiber Failure (FF)
        # ---------------------------------------------------------
        # Puck Fiber Tension/Compression map directly to standard Fiber Enums.

        magnification_term = (self.v12 / self.E1) * self.m_sigF * s2

        if s1 >= 0:
            val = (e1 + magnification_term) / self.eps_1T
            if val >= 1:
                failures.append(FailureMode.FIBER_TENSION)
        else:
            val = abs(e1 + magnification_term) / abs(self.eps_1C)
            if val >= 1:
                failures.append(FailureMode.FIBER_COMPRESSION)

        # ---------------------------------------------------------
        # 2. Inter-Fiber Failure (IFF)
        # ---------------------------------------------------------
        # Calculate Weakening Factor (eta_w1)
        s1_limit = self.Xt if s1 >= 0 else self.Xc
        weakening = abs(s1 / s1_limit) * self.weakening_coeff

        if s2 >= 0:
            # --- IFF Mode A (Tension) ---
            # Maps to MATRIX_TENSION
            coeff = 1 - self.p12_plus * (self.Yt / self.S)
            term_shear = t12 / self.S
            term_stress = s2 / self.Yt

            root = np.sqrt(term_shear**2 + coeff**2 * term_stress**2)
            fi_a = root + self.p12_plus * (s2 / self.S) + weakening

            if fi_a >= 1:
                failures.append(FailureMode.MATRIX_TENSION)

        else:
            # --- IFF Modes B & C (Compression) ---
            # Both map to MATRIX_COMPRESSION, as the physical result
            # is the matrix crushing/shearing, losing transverse stiffness.

            ratio_sigma_tau = abs(s2 / t12) if abs(t12) > 1e-9 else 1e9
            turning_point = self.R_tt_A / abs(self.tau_12_c)

            if 0 <= ratio_sigma_tau <= turning_point:
                # Mode B
                root = np.sqrt(t12**2 + (self.p12_minus * s2) ** 2)
                fi_b = (1 / self.S) * (root + self.p12_minus * s2) + weakening
                if fi_b >= 1:
                    failures.append(FailureMode.MATRIX_COMPRESSION)
            else:
                # Mode C
                term1 = (t12 / (2 * (1 + self.p23_minus) * self.S)) ** 2
                term2 = (s2 / self.Yc) ** 2
                fi_c = ((term1 + term2) * (self.Yc / -s2)) + weakening
                if fi_c >= 1:
                    failures.append(FailureMode.MATRIX_COMPRESSION)

        return failures

    def get_effort(self, local_stress, local_strain) -> float:
        s1, s2, t12 = local_stress
        e1 = local_strain[0]

        # 1. Fiber Failure (FF)
        magnification_term = (self.v12 / self.E1) * self.m_sigF * s2
        if s1 >= 0:
            fi_ff = (e1 + magnification_term) / self.eps_1T
        else:
            fi_ff = abs(e1 + magnification_term) / abs(self.eps_1C)

        # 2. Inter-Fiber Failure (IFF)
        s1_limit = self.Xt if s1 >= 0 else self.Xc
        weakening = abs(s1 / s1_limit) * self.weakening_coeff

        if s2 >= 0:
            # Mode A
            coeff = 1 - self.p12_plus * (self.Yt / self.S)
            term_shear = t12 / self.S
            term_stress = s2 / self.Yt
            root = np.sqrt(term_shear**2 + coeff**2 * term_stress**2)
            fi_iff = root + self.p12_plus * (s2 / self.S) + weakening
        else:
            ratio_sigma_tau = abs(s2 / t12) if abs(t12) > 1e-9 else 1e9
            turning_point = self.R_tt_A / abs(self.tau_12_c)

            if 0 <= ratio_sigma_tau <= turning_point:
                # Mode B
                root = np.sqrt(t12**2 + (self.p12_minus * s2) ** 2)
                fi_iff = (1 / self.S) * (root + self.p12_minus * s2) + weakening
            else:
                # Mode C
                term1 = (t12 / (2 * (1 + self.p23_minus) * self.S)) ** 2
                term2 = (s2 / self.Yc) ** 2
                fi_iff = ((term1 + term2) * (self.Yc / -s2)) + weakening

        return max(fi_ff, fi_iff)


# PUCK, HASHIN, CHRISTENSEN, LARC05
