import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from classical_laminate_theory.failuremodels import FailureModel


class Lamina:
    """
    A single composite ply.
    Uses @property for stiffness matrices to support Progressive Failure Analysis (PFA).
    When E1/E2 degrade, Qmat and Qbarmat update automatically.
    """

    def __init__(
        self,
        angle: float,
        E1: float,
        E2: float,
        G12: float,
        v12: float,
        t: float,
        failure_model: FailureModel,
    ):
        # 1. Material Inputs (State variables that might change in PFA)
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.v12 = v12
        self.t = t
        self.failure_model = failure_model

        # 2. Geometric Inputs (assumed static for PFA)
        self.angle = np.radians(angle)
        self.m = np.cos(self.angle)
        self.n = np.sin(self.angle)

        # 3. State Flags
        self.failed = False

    @property
    def global_local_strain_transform(self):
        """
        Transformation Matrix T_epsilon (Global -> Local) for ENGINEERING strains.
        Differs from stress transform in the shear row (row 3).
        """
        m, n = self.m, self.n
        return np.array(
            [
                [m ** 2, n ** 2, m * n],  # mn instead of 2mn
                [n ** 2, m ** 2, -m * n],  # -mn instead of -2mn
                [-2 * m * n, 2 * m * n, m ** 2 - n ** 2],  # 2mn factor here is critical
            ]
        )

    @property
    def v21(self):
        """Minor Poisson's ratio."""
        return self.v12 * self.E2 / self.E1

    @property
    def Qmat(self):
        """Reduced Stiffness Matrix (Local). Computed on-the-fly."""
        denom = 1 - (self.v12 * self.v21)
        Q11 = self.E1 / denom
        Q12 = (self.v12 * self.E2) / denom
        Q22 = self.E2 / denom
        Q66 = self.G12

        return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    @property
    def Qbarmat(self):
        """Transformed Stiffness Matrix (Global). Computed on-the-fly."""
        Q = self.Qmat
        Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

        m, n = self.m, self.n
        m2, n2 = m**2, n**2
        m4, n4 = m**4, n**4

        # Standard CLT Transformations
        Qxx = (m4 * Q11) + (n4 * Q22) + (2 * m2 * n2 * Q12) + (4 * m2 * n2 * Q66)
        Qyy = (n4 * Q11) + (m4 * Q22) + (2 * m2 * n2 * Q12) + (4 * m2 * n2 * Q66)
        Qxy = (
            (m2 * n2 * Q11) + (m2 * n2 * Q22) + ((m4 + n4) * Q12) - (4 * m2 * n2 * Q66)
        )
        Qxs = (
            (m**3 * n * Q11)
            - (m * n**3 * Q22)
            - (m * n * (m2 - n2) * Q12)
            - (2 * m * n * (m2 - n2) * Q66)
        )
        Qys = (
            (n**3 * m * Q11)
            - (n * m**3 * Q22)
            + (m * n * (m2 - n2) * Q12)
            + (2 * m * n * (m2 - n2) * Q66)
        )
        Qss = (
            (m2 * n2 * Q11)
            + (m2 * n2 * Q22)
            - (2 * m2 * n2 * Q12)
            + ((m2 - n2) ** 2 * Q66)
        )

        return np.array([[Qxx, Qxy, Qxs], [Qxy, Qyy, Qys], [Qxs, Qys, Qss]])

    @property
    def global_local_transform(self):
        """Transformation Matrix T (Global -> Local)."""
        m, n = self.m, self.n
        return np.array(
            [
                [m**2, n**2, 2 * m * n],
                [n**2, m**2, -2 * m * n],
                [-m * n, m * n, m**2 - n**2],
            ]
        )


class Laminate:
    def __init__(
        self,
        plies: tuple[Lamina, ...],
        load=np.array([0, 0, 0, 0, 0, 0]),
    ):
        """
        Composite laminate class supporting Progressive Failure.
        All attributes are explicitly defined here to satisfy linters.
        """
        self.plies = plies
        self.load = load
        self.n_plies = len(plies)

        # --- 1. Geometry Attributes (Defined before calculation) ---
        self.lamina_boundaries = np.zeros(self.n_plies + 1)
        self.lamina_midplanes = np.zeros(self.n_plies)

        # --- 2. Stiffness Attributes (Initialized to Zero/None) ---
        # Cached list of ply stiffnesses (refreshed in update_stiffness)
        self.Qbar_matrices = None
        # Transformation matrices (Static)
        self.T_stress_matrices = np.array([ply.global_local_transform for ply in self.plies])
        self.T_strain_matrices = np.array([ply.global_local_strain_transform for ply in self.plies])

        # ABD Matrices
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.ABD = np.zeros((6, 6))
        self.abd = np.zeros((6, 6))  # Inverse ABD

        # --- 3. Stress/Strain Attributes (Initialized to Zero) ---
        # Shape: (n_plies, 3)
        self.global_strains = np.zeros((self.n_plies, 3))
        self.global_stresses = np.zeros((self.n_plies, 3))
        self.local_strains = np.zeros((self.n_plies, 3))
        self.local_stresses = np.zeros((self.n_plies, 3))

        # --- 4. Trigger Initial Calculations ---
        self.get_z_positions()
        self.update_stiffness()
        self.get_stress_strain()

    @classmethod
    def from_layup(
            cls,
            layup_string: str,
            E1: float,
            E2: float,
            G12: float,
            v12: float,
            t: float,
            failure_model: FailureModel,
            load=None
    ):
        """
        Factory method to create a Laminate from a string and uniform material properties.

        Args:
            layup_string: e.g. "[0/90]_s" or "[0, 45, -45, 90]_2s"
            E1, E2, G12, v12: Elastic constants
            t: Ply thickness
            failure_model: Instance of FailureModel (Hashin, TsaiHill, etc.)
            load: Optional initial load vector (1x6)

        Returns:
            Laminate instance
        """
        # Local import to prevent circular dependency with helpers.py
        from .helpers import parse_layup_string

        if load is None:
            load = np.zeros(6)

        # Get list of angles (e.g., [0, 90, 90, 0])
        angles = parse_layup_string(layup_string)

        plies = []
        for angle in angles:
            # Create identical material for every angle
            plies.append(Lamina(
                angle=angle,
                E1=E1,
                E2=E2,
                G12=G12,
                v12=v12,
                t=t,
                failure_model=failure_model
            ))

        # Return new instance of class (Laminate)
        return cls(tuple(plies), load)

    def get_z_positions(self):
        """Calculate z-coordinates of ply interfaces and midplanes."""
        ply_thickness = np.array([ply.t for ply in self.plies])
        z = np.hstack(([0], np.cumsum(ply_thickness)))
        midplane_shift = 0.5 * (z[0] + z[-1])
        self.lamina_boundaries = z - midplane_shift
        self.lamina_midplanes = self.lamina_boundaries[:-1] + 0.5 * ply_thickness

    def update_stiffness(self):
        """
        Recalculates A, B, D matrices.
        Must be called if ply properties (E1, E2) change.
        """
        # Reset Accumulators
        self.A.fill(0)
        self.B.fill(0)
        self.D.fill(0)

        # Refresh stiffness from plies (triggers @property re-computation)
        self.Qbar_matrices = np.array([ply.Qbarmat for ply in self.plies])

        z_bounds = self.lamina_boundaries

        for i in range(self.n_plies):
            bottom_z, top_z = z_bounds[i], z_bounds[i + 1]
            dz = top_z - bottom_z
            dz2_half = (top_z**2 - bottom_z**2) / 2.0
            dz3_third = (top_z**3 - bottom_z**3) / 3.0

            Qbar = self.Qbar_matrices[i]

            self.A += Qbar * dz
            self.B += Qbar * dz2_half
            self.D += Qbar * dz3_third

        self.ABD = np.block([[self.A, self.B], [self.B.T, self.D]])

        det = np.linalg.det(self.ABD)
        if np.abs(det) > 1e-12:
            self.abd = np.linalg.inv(self.ABD)
            return True  # Signal success
        else:
            # Matrix is singular (part is broken)
            # We don't raise an error; we just don't calculate 'abd'
            self.abd = np.zeros_like(self.ABD)
            return False  # Signal failure

    def get_stress_strain(self):
        """Compute stresses based on current Load and Stiffness."""
        midplane_deformation = np.linalg.solve(self.ABD, self.load)

        midplane_strains = midplane_deformation[:3]
        curvatures = midplane_deformation[3:]

        # Global Strains: epsilon_z = epsilon_0 + z * kappa
        self.global_strains = (
            midplane_strains + self.lamina_midplanes[:, None] * curvatures
        )

        # Global Stresses: sigma = Qbar * epsilon
        # Einsum: ply (n), row (i), col (j)
        self.global_stresses = np.einsum(
            "nij,nj->ni", self.Qbar_matrices, self.global_strains
        )

        # Local Strains: Use T_strain_matrices
        self.local_strains = np.einsum(
            "nij,nj->ni", self.T_strain_matrices, self.global_strains
        )

        # Local Stresses: Use T_stress_matrices (formerly self.T_matrices)
        self.local_stresses = np.einsum(
            "nij,nj->ni", self.T_stress_matrices, self.global_stresses
        )

    def apply_load(self, load):
        """Update load and re-solve stresses without full stiffness rebuild."""
        self.load = load
        self.get_stress_strain()

    # (Keep your _plot_distribution, global_stress_graph, etc. methods here)
    # They will work perfectly with this structure.

    def _plot_distribution(
        self,
        values,
        labels,
        title,
    ):
        """
        General function to plot stress/strain distributions with fiber angle visualization.

        Args:
            values: The values to plot (stresses or strains)
            labels: Labels for each axis
            title: Plot title
        """
        sns.set_style("darkgrid")
        sns.set_palette("deep")

        z_bounds = self.lamina_boundaries
        colors = sns.color_palette("deep", 3)

        # Create the main figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, dpi=100)
        fig.subplots_adjust(
            right=0.82  # Adjusted to make more space for the right-hand labels
        )

        # Plot the main data in the subplots
        for i, ax in enumerate(axes):
            for j in range(len(self.plies)):
                z_lower, z_upper = z_bounds[j], z_bounds[j + 1]
                val_lower, val_upper = values[j, i], values[j, i]

                # Plot the vertical line for the ply
                ax.plot(
                    [val_lower, val_upper],
                    [z_lower, z_upper],
                    color=colors[i],
                    linestyle="-",
                    linewidth=2,
                )

                # Add horizontal lines at ply breaks (now solid)
                if j < len(self.plies) - 1:
                    ax.plot(
                        [val_upper, values[j + 1, i]],
                        [z_upper, z_upper],
                        color=colors[i],
                        linestyle="-",
                        linewidth=2,
                    )
                    # Add solid horizontal line at the break
                    ax.axhline(
                        y=z_upper,
                        color="gray",
                        linestyle="-",  # Changed to solid
                        alpha=0.3,
                        linewidth=0.8,
                    )

            ax.axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=1.2)
            ax.set_xlabel(labels[i], fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)

        # Set y-label only for the first subplot
        axes[0].set_ylabel("Z position (m)", fontsize=12, fontweight="bold")

        # Add ply angles to the far right of the figure
        for j in range(len(self.plies)):
            z_lower, z_upper = z_bounds[j], z_bounds[j + 1]
            mid_z = float((z_lower + z_upper) / 2)
            ply_angle = np.degrees(self.plies[j].angle)

            # Place ply angle labels on the far right of the figure
            axes[0].annotate(
                text=f"{ply_angle:.2f}°",
                xy=(3.4, mid_z),  # Position just outside the right edge of the plot
                xycoords=(
                    "axes fraction",
                    "data",
                ),  # Use axes fraction for x, data for y
                fontsize=10,
                color="k",
                ha="left",
                va="center",
                xytext=(10, 0),  # Offset in points
                textcoords="offset points",
            )

        # Add a label for the ply angles on the far right of the figure
        axes[2].text(
            x=3.62,  # Position just outside the right edge of the plot
            y=float((z_bounds[0] + z_bounds[-1]) / 2),
            s="Ply angle (degrees)",
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
            transform=axes[
                0
            ].get_yaxis_transform(),  # Use y-axis transform for positioning
        )

        # Add heavy horizontal lines at the top and bottom of the laminate
        for ax in axes:
            ax.axhline(
                y=z_bounds[0], color="gray", linestyle="-", linewidth=2, alpha=0.3
            )
            ax.axhline(
                y=z_bounds[-1], color="gray", linestyle="-", linewidth=2, alpha=0.3
            )
            # When you hatch to the edge it will expand the y axis (for some reason), so store the current bounds and
            # reset them after running the hatching operations
            ax_bounds = ax.get_ylim()
            ax.axhspan(
                ymin=ax_bounds[0],  # Minimum y-limit of the plot
                ymax=z_bounds[0],  # Minimum z bound
                color="gray",
                alpha=0.1,  # Light transparency
                hatch="///",  # Hatching pattern
                linewidth=0.5,
            )
            # Hatching above the maximum z bound
            ax.axhspan(
                ymin=z_bounds[-1],  # Maximum z bound
                ymax=ax_bounds[1],  # Maximum y-limit of the plot
                color="gray",
                alpha=0.1,  # Light transparency
                hatch="///",  # Hatching pattern
                linewidth=0.5,
            )
            ax.set_ylim(ax_bounds)

        fig.suptitle(title, fontweight="bold")
        plt.show()

    @property
    def global_stress_graph(self):
        """Plot stress distribution through laminate thickness."""
        return self._plot_distribution(
            self.global_stresses * 1e-6,
            [r"$\sigma_{xx}$ (MPa)", r"$\sigma_{yy}$ (MPa)", r"$\tau_{xy}$ (MPa)"],
            "Global Stress Distribution in the Laminate",
        )

    @property
    def global_strain_graph(self):
        """Plot strain distribution through laminate thickness."""
        return self._plot_distribution(
            self.global_strains,
            [r"$\varepsilon_{xx}$", r"$\varepsilon_{yy}$", r"$\gamma_{xy}$"],
            "Global Strain Distribution in the Laminate",
        )

    @property
    def local_stress_graph(self):
        """Plot stress distribution through laminate thickness."""
        return self._plot_distribution(
            self.local_stresses * 1e-6,
            [r"$\sigma_{11}$ (MPa)", r"$\sigma_{22}$ (MPa)", r"$\tau_{12}$ (MPa)"],
            "Local Stress Distribution in the Laminate",
        )

    @property
    def local_strain_graph(self):
        """Plot strain distribution through laminate thickness."""
        return self._plot_distribution(
            self.local_strains,
            [r"$\varepsilon_{11}$", r"$\varepsilon_{22}$", r"$\gamma_{12}$"],
            "Local Strain Distribution in the Laminate",
        )
