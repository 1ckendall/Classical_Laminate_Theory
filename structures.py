import numpy as np
from abc import ABC, abstractmethod
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Annotated


class Lamina:
    """A single composite lamina. Base class for laminae."""

    def __init__(
        self,
        angle: float,
        E1: float,
        E2: float,
        G12: float,
        v12: float,
        t: float,
    ):
        """
        Initialise a composite lamina with given properties.
        Args:
            angle: Angle of fibres (deg).
            E1: E-modulus along the fibres (Pa)
            E2: E-modulus across the fibres (Pa)
            G12: Shear modulus (Pa)
            v12: Major Poisson ratio
            t: Ply Thickness (m)
        """
        self.angle = np.radians(angle)  # angle of fibres
        self.m = np.cos(self.angle)
        self.n = np.sin(self.angle)
        self.E1 = E1  # E along fibres
        self.E2 = E2  # E across fibres
        self.G12 = G12  # Shear modulus
        self.v12 = v12  # Major Poisson
        self.v21 = self.v12 * self.E2 / self.E1  # Minor Poisson
        self.t = t  # Thickness
        self.failed = False

        # Initialise reduced stiffness matrix Q parallel to the fibres of the lamina
        self.Q = 1 - (self.v12 * self.v21)
        self.Q11 = self.E1 / self.Q
        self.Q12 = self.v12 * self.E2 / self.Q
        self.Q22 = self.E2 / self.Q
        self.Q66 = self.G12
        # Disable Black reformatting for the next line
        # fmt: off
        self.Qmat = np.array(
            [[self.Q11, self.Q12,        0],
             [self.Q12, self.Q22,        0],
             [0,        0,        self.Q66]]
        )
        # fmt: on

        # Initialise reduced stiffness matrix Qbar matrix parallel to the global coordinate system
        self.Qxx = (
            self.m**4 * self.Q11
            + self.n**4 * self.Q22
            + 2 * self.m**2 * self.n**2 * self.Q12
            + 4 * self.m**2 * self.n**2 * self.Q66
        )
        self.Qyy = (
            self.n**4 * self.Q11
            + self.m**4 * self.Q22
            + 2 * self.m**2 * self.n**2 * self.Q12
            + 4 * self.m**2 * self.n**2 * self.Q66
        )
        self.Qxy = (
            self.m**2 * self.n**2 * self.Q11
            + self.m**2 * self.n**2 * self.Q22
            + (self.m**4 + self.n**4) * self.Q12
            - 4 * self.m**2 * self.n**2 * self.Q66
        )
        self.Qxs = (
            self.m**3 * self.n * self.Q11
            - self.m * self.n**3 * self.Q22
            - self.m * self.n * (self.m**2 - self.n**2) * self.Q12
            - 2 * self.m * self.n * (self.m**2 - self.n**2) * self.Q66
        )
        self.Qys = (
            self.n**3 * self.m * self.Q11
            - self.n * self.m**3 * self.Q22
            + self.m * self.n * (self.m**2 - self.n**2) * self.Q12
            + 2 * self.m * self.n * (self.m**2 - self.n**2) * self.Q66
        )
        self.Qss = (
            self.m**2 * self.n**2 * self.Q11
            + self.m**2 * self.n**2 * self.Q22
            - 2 * self.m**2 * self.n**2 * self.Q12
            + (self.m**2 - self.n**2) ** 2 * self.Q66
        )
        self.Qbarmat = np.array(
            [
                [self.Qxx, self.Qxy, self.Qxs],
                [self.Qxy, self.Qyy, self.Qys],
                [self.Qxs, self.Qys, self.Qss],
            ]
        )

    @abstractmethod
    def failure(self, stress, strain):
        return False


class TsaiHillLamina(Lamina):
    def __init__(
        self,
        angle: float,
        E1: float,
        E2: float,
        G12: float,
        v12: float,
        t: float,
        X11: float = 0,
        X22: float = 0,
        S12: float = 0,
    ):
        """
        A lamina using the Tsai-Hill failure criterion
        Args:
            angle: Angle of fibres (deg).
            E1: E-modulus along the fibres (Pa)
            E2: E-modulus across the fibres (Pa)
            G12: Shear modulus (Pa)
            v12: Major Poisson ratio
            t: Ply Thickness (m)
            X11: Allowable strength of the ply in the longitudinal direction (0° direction)
            X22: Allowable strength of the ply in the transversal direction (90° direction)
            S12: Allowable in-plane shear strength of the ply between the longitudinal and the
            transversal directions
        """
        super().__init__(angle, E1, E2, G12, v12, t)
        self.X11 = X11
        self.X22 = X22
        self.S12 = S12

    def failure(self, stress, strain):
        criterion = (
            (stress[0] / self.X11) ** 2
            - (stress[0] * stress[1]) / (self.X11**2)
            + (stress[1] / self.X22) ** 2
            + (stress[2] / self.S12) ** 2
        )
        if criterion < 1:
            return False
        return True


# PUCK, HASHIN, CHRISTENSEN, LARC05


class PuckLamina(Lamina):
    def __init__(
        self, angle: float, E1: float, E2: float, G12: float, v12: float, t: float
    ):
        super().__init__(angle, E1, E2, G12, v12, t)


class Laminate:
    def __init__(self, plies: tuple[Lamina,...], load: Annotated[np.ndarray, (6,)]):
        self.plies = plies
        self.n = len(plies)
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.ABD = np.zeros((6, 6))
        self.abd = np.zeros((6, 6))
        self.load = load
        self.stresses = np.zeros([len(plies), 3])
        self.strains = np.zeros([len(plies), 3])
        self.get_z_positions()
        self.get_abd_matrices()
        self.get_stress_strain()

    def get_z_positions(self):
        """Calculate the boundary and midplane z-positions for each ply."""
        ply_thickness = np.array([ply.t for ply in self.plies])
        z = np.hstack(([0], np.cumsum(ply_thickness)))
        midplane_shift = 0.5 * (z[0] + z[-1])
        self.lamina_boundaries = z - midplane_shift
        self.lamina_midplanes = self.lamina_boundaries[:-1] + 0.5 * ply_thickness

    def get_abd_matrices(self):
        """Efficiently compute A, B, and D matrices and assemble ABD."""
        z_bounds = self.lamina_boundaries
        for i, ply in enumerate(self.plies):
            bottom_z, top_z = z_bounds[i], z_bounds[i + 1]
            dz = top_z - bottom_z
            dz2_half = (top_z**2 - bottom_z**2) / 2
            dz3_third = (top_z**3 - bottom_z**3) / 3

            self.A += ply.Qbarmat * dz
            self.B += ply.Qbarmat * dz2_half
            self.D += ply.Qbarmat * dz3_third

        self.ABD = np.block([[self.A, self.B], [self.B.T, self.D]])
        det_abd = np.linalg.det(self.ABD)
        if det_abd != 0:
            self.abd = np.linalg.inv(self.ABD)
        else:
            raise ValueError("ABD matrix is singular; check input properties.")

    def get_stress_strain(self):
        """Compute laminate strains and stresses for each ply more efficiently."""
        strain_midplane = np.linalg.solve(self.ABD, self.load)
        self.strains = (
            strain_midplane[:3] + np.outer(self.lamina_midplanes, strain_midplane[3:])
        )
        self.stresses = np.einsum("ijk,ik->ij", np.array([ply.Qbarmat for ply in self.plies]), self.strains)

    def update_load(self, Nx=0, Ny=0, Ns=0, Mx=0, My=0, Ms=0):
        """Update the applied load and recompute stress/strain, with error handling."""
        try:
            self.load = np.array([Nx, Ny, Ns, Mx, My, Ms], dtype=float)
            self.get_stress_strain()
        except ValueError as e:
            raise ValueError("Invalid load values or singular ABD matrix.") from e

    def _plot_distribution(self, values, labels, title):
        """General function to plot stress/strain distributions."""
        z_bounds = self.lamina_boundaries
        colors = ["b", "r", "g"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

        for i, ax in enumerate(axes):
            for j in range(len(self.plies)):
                z_lower, z_upper = z_bounds[j], z_bounds[j + 1]
                val_lower, val_upper = values[j, i], values[j, i]
                ax.plot([val_lower, val_upper], [z_lower, z_upper], color=colors[i], linestyle="-")
                if j < len(self.plies) - 1:
                    ax.plot([val_upper, values[j + 1, i]], [z_upper, z_upper], color=colors[i], linestyle="-")

            ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
            ax.set_xlabel(labels[i])
            ax.grid(True)

        axes[0].set_ylabel("Laminate Thickness Position (m)")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    @property
    def stress_graph(self):
        """Plot stress distribution through laminate thickness."""
        return self._plot_distribution(
            self.stresses, [r"$\sigma_x$ (MPa)", r"$\sigma_y$ (MPa)", r"$\tau_{xy}$ (MPa)"], "Stress Distribution in the Laminate"
        )

    @property
    def strain_graph(self):
        """Plot strain distribution through laminate thickness."""
        return self._plot_distribution(
            self.strains, [r"$\varepsilon_x$", r"$\varepsilon_y$", r"$\gamma_{xy}$"], "Strain Distribution in the Laminate"
        )


if __name__ == "__main__":
    l1 = Lamina(0, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l2 = Lamina(90, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l3 = Lamina(0, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l4 = Lamina(90, 200e9, 50e9, 100e6, 0.4, 0.1e-3)
    l5 = Lamina(0, 200e9, 50e9, 100e6, 0.4, 0.1e-3)

    L2 = Laminate(plies=(l1, l2, l3, l4, l5), Nx=1e6)
    print(L2.stresses)
    print()
    print(L2.lamina_boundaries)
    L2.stress_graph()
